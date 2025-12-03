import os
import json
import shutil
import argparse
from multiprocessing import Process, Queue
import pickle
import copy
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
from vllm import LLM, SamplingParams

from utils import PCARepReader


class AdaptiveController:

    def __init__(self, 
                 model, 
                 batch_size,
                 monitor_layers, 
                 control_layers, 
                 control_vectors, 
                 coeff_step_size=2, 
                 min_coeff=-4, 
                 max_coeff=8,
                 device="cuda",
                 eos_token=151643,
                 # For sliding window adjustment
                 sliding_window_size=8,
                 sliding_window_threshold=2):
        
        self.num_layers = len(model.model.layers)

        self.model = model
        self.lm_head = model.lm_head.weight.detach().clone()
        self.norm = copy.deepcopy(model.model.norm)

        # (L, D)
        self.control_vectors = control_vectors.type(torch.bfloat16).to(device)

        self.batch_size  = batch_size
        self.coeff_step_size = coeff_step_size
        self.min_coeff = min_coeff
        self.max_coeff = max_coeff
        self.coeffs = torch.tensor([self.min_coeff]*batch_size).to(device)
        
        # Register control hooks first, then monitor hooks
        self.control_hooks = []
        for layer_id, layer in enumerate(control_layers):
            hook = self.model.model.layers[layer].register_forward_hook(partial(self.controll_fn, layer_id=layer_id))
            self.control_hooks.append(hook)

        self.monitor_hooks = []
        self.monitor_outputs = []
        self.monitor_layers = monitor_layers
        for layer in monitor_layers:
            hook = self.model.model.layers[layer].register_forward_hook(self.monitor_fn)
            self.monitor_hooks.append(hook)
        
        # Activate at the last layer
        self.adjust_coeff_hook = self.model.model.layers[-1].register_forward_hook(self.adjust_coeff_fn)

        ## Sliding window params
        self.sliding_window_js_divs = torch.zeros((batch_size, sliding_window_size), dtype=torch.bfloat16, device=device)
        self.sliding_window_size = sliding_window_size
        self.sliding_window_threshold = sliding_window_threshold
        self.sliding_window_cnt  = 0

        # Activate at last
        self.eos_token = eos_token
        self.filter_terminated_seq_hook = self.model.sampler.register_forward_hook(self.filter_terminated_seq)

        self.last_idxs = torch.zeros(self.batch_size).to(device)

    @staticmethod
    def get_last_idxs(input, output):
        # Output Shape:
        # Start position: Tuple((B * L(Prompt), D), (B * L(Prompt), D))
        # Afterwards:     Tuple((B, D), (B, D))
        def find_sequence_ends(vector):
            diffs = vector[1:] - vector[:-1]
            breaks = torch.where(diffs != 1)[0]
            ends = torch.cat([breaks, torch.tensor([len(vector)-1], device=vector.device)])
            return ends

        positions = input[0].detach().clone()
        last_idxs = find_sequence_ends(positions)

        return last_idxs

    def get_norm(self, x, residual):
        h, _ = self.norm(x, residual)
        return h

    def get_logits(self, hidden_state):
        logits = hidden_state @ self.lm_head.T
        return logits
    
    def get_layer_differences(self, layer_outputs):

        layer_logits = torch.stack(layer_outputs)

        # (Num Layers, Batch Size, Hidden Size)
        premature_logits = layer_logits[:-1]
        mature_logits    = layer_logits[-1]

        softmax_premature_hidden_states = F.softmax(premature_logits, dim=-1)
        softmax_mature_hidden_states = F.softmax(mature_logits, dim=-1)

        M = 0.5 * (softmax_mature_hidden_states[None, ...] + softmax_premature_hidden_states)

        log_softmax_premature_hidden_states = F.log_softmax(premature_logits, dim=-1)
        log_softmax_mature_hidden_states = F.log_softmax(mature_logits, dim=-1)

        kl1 = F.kl_div(log_softmax_premature_hidden_states, M, reduction='none').mean(-1)
        kl2 = F.kl_div(log_softmax_mature_hidden_states[None, ...], M, reduction='none').mean(-1)

        # (Num Premaure Layers, Batch Size)
        js_divs = 0.5 * (kl1 + kl2)
        avg_js_div = torch.mean(js_divs, dim=0)

        return avg_js_div

    def controll_fn(self, module, input, output, layer_id):

        self.last_idxs = self.get_last_idxs(input, output)
        h, r = output
        h[self.last_idxs, :] += (self.coeffs.view(-1, 1) * self.control_vectors[layer_id].view(1, -1))

        new_output = (h, r)
        return new_output

    def monitor_fn(self, module, input, output):

        self.last_idxs = self.get_last_idxs(input, output)
        x, residual = output[0][self.last_idxs, :], output[1][self.last_idxs, :]
        hidden_state = self.get_norm(x, residual)
        layer_logits = self.get_logits(hidden_state)
        self.monitor_outputs.append(layer_logits)
        return output

    def sliding_window_adjust(self, js_div, accelerate_coeffs, stop_coeffs):
        if self.sliding_window_cnt < self.sliding_window_size:
            self.sliding_window_js_divs[:, self.sliding_window_cnt] = js_div
            self.sliding_window_cnt += 1
            return self.coeffs
        else:
            # (B,)
            avg_js_div = torch.mean(self.sliding_window_js_divs,dim=1)
            std_js_div = torch.std (self.sliding_window_js_divs, dim=1)
            stop_threshold = avg_js_div + self.sliding_window_threshold * std_js_div
            _sliding_window_js_divs = self.sliding_window_js_divs.clone()
            
            self.sliding_window_js_divs[:, :-1] = _sliding_window_js_divs[:, 1:]
            self.sliding_window_js_divs[:, -1]  = js_div

            return torch.where(js_div < stop_threshold, accelerate_coeffs, stop_coeffs)


    def adjust_coeff_fn(self, module, input, output):

        self.last_idxs = self.get_last_idxs(input, output)
               
        assert len(self.monitor_outputs) == len(self.monitor_layers)

        x, residual = output[0][self.last_idxs, :], output[1][self.last_idxs, :]

        hidden_state = self.get_norm(x, residual)
        layer_logits = self.get_logits(hidden_state)

        js_div = self.get_layer_differences(self.monitor_outputs + [layer_logits]) * 10e5


        stop_coeffs       = torch.clamp(self.coeffs - self.coeff_step_size, min=self.min_coeff, max=self.min_coeff)
        accelerate_coeffs = torch.clamp(self.coeffs + self.coeff_step_size, min=self.min_coeff, max=self.max_coeff)
        self.coeffs = self.sliding_window_adjust(js_div, accelerate_coeffs, stop_coeffs)

        # Empty the monitor outputs
        self.monitor_outputs = []

        return output

    def filter_terminated_seq(self, module, input, output):
        if len(self.coeffs) > 1:
            # Filter out the sequences that are terminated
            output_ids = [o.samples[0].output_token for o in output.outputs]
            assert len(output_ids) == len(self.coeffs)

            next_valid_seq_ids = [idx for idx in range(len(output_ids)) if output_ids[idx] != self.eos_token]

            self.coeffs = self.coeffs[next_valid_seq_ids]
            self.sliding_window_js_divs = self.sliding_window_js_divs[next_valid_seq_ids]

        return output
        
    def remove_hooks(self, ):
        for h in self.control_hooks:
            h.remove()
        for h in self.monitor_hooks:
            h.remove()
        self.adjust_coeff_hook.remove()
        self.filter_terminated_seq_hook.remove()
        return
    
    def reset_coeffs(self, device="cuda"):

        self.coeffs = torch.tensor([self.min_coeff]*self.batch_size).to(device)
        
        self.skip_token_cnt = 0
        self.sliding_window_js_divs = torch.zeros((self.batch_size, self.sliding_window_size), dtype=torch.bfloat16, device=device)
        self.sliding_window_cnt  = 0

        self.history_coeffs = []
        self.history_js_divs = []
        self.history_output_ids = []


class RepController:

    def __init__(self, module, control_vector):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.control_vector = control_vector
        self.control_times = 0
    
    def get_last_idxs(self, input, output):
        # Output Shape:
        # Start position: Tuple((B * L(Prompt), D), (B * L(Prompt), D))
        # Afterwards:     Tuple((B, D), (B, D))

        positions = input[0]
        # Test if it is at the start position
        if 0 in positions:
            last_idxs = torch.where(positions == 0)[0][1:]
            if len(last_idxs) > 0:
                last_idxs = last_idxs-1
            
            last_idxs = torch.cat([last_idxs, torch.tensor([len(positions)-1],dtype=last_idxs.dtype, device=last_idxs.device)])
        else:
            last_idxs = list(range(output[0].shape[0]))
            
        return last_idxs
    
    def hook_fn(self, module, input, output):

        # Constant Steering
        last_idxs = self.get_last_idxs(input, output)
        
        h, r = output
        h[last_idxs, :] += self.control_vector
        new_output = (h, r)

        return new_output

    def close(self):
        self.hook.remove()


def set_repcontroller(model, target_layer, rep_reader, 
                      control_coeff=4.0,
                      device="cuda"):
    
    if isinstance(target_layer, int):
        target_layer = [target_layer]
    
    hooks = []
    for layer in target_layer:
        control_vector = torch.tensor(control_coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).type(torch.bfloat16).to(device)
        hook = RepController(model.model.layers[layer], control_vector)
        hooks.append(hook)
    
    return hooks


def close_repcontroller(hooks):
    for hook in hooks:
        hook.close()


def inference(args, data_queue: Queue, result_save_path, process_id):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(process_id)

    chat_template = "<｜User｜>{question}<｜Assistant｜><think>\n"

    llm = LLM(args.model_name_or_path, 
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            device="auto",
            enforce_eager=True
        )

    if args.intervention_coeff != 0 and args.intervention_type == "constant":
        direction_finder = pickle.load(open(args.interventor_ckpt, "rb"))
        hooks = set_repcontroller(llm.llm_engine.model_executor.driver_worker.model_runner.model, 
                                    list(range(args.intervention_start_layer, args.intervention_end_layer+1,)), 
                                    direction_finder, 
                                    control_coeff=args.intervention_coeff, 
                                    device=f"cuda")
        
    elif args.intervention_type == "adaptive":
        direction_finder = pickle.load(open(args.interventor_ckpt, "rb"))

        control_layers=list(range(args.intervention_start_layer, args.intervention_end_layer+1,))
        control_vectors = []
        for layer in control_layers:
            control_vectors.append(torch.tensor(direction_finder.directions[layer] * direction_finder.direction_signs[layer])[0])
        control_vectors = torch.stack(control_vectors)

        ada_ctrl = AdaptiveController(
            model=llm.llm_engine.model_executor.driver_worker.model_runner.model,
            batch_size=args.n,
            monitor_layers=list(range(
                args.adaptive_controller_monitor_start_layer, 
                args.adaptive_controller_monitor_end_layer+1, 
                args.adaptive_controller_monitor_layer_interval
            )),
            control_layers=control_layers,
            control_vectors=control_vectors,
            min_coeff=args.adaptive_controller_min_coeff,
            max_coeff=args.adaptive_controller_max_coeff,
            sliding_window_size=args.adaptive_controller_sliding_window_size,
            sliding_window_threshold=args.adaptive_controller_sliding_window_threshold,
        )


    result_data = []

    show_example = True

    while not data_queue.empty():
        try:
            inference_data = data_queue.get_nowait()
        except Exception as e:
            print("Exception occurred:", e)
            continue

        if args.intervention_type == "adaptive":
            ada_ctrl.reset_coeffs()
        
        question = inference_data["user_prompt"]
        input_prompt = chat_template.format(question=question)
        if args.enable_fast_thinking:
            input_prompt = input_prompt + "To"

        if show_example:
            print(f"Example input: {input_prompt}")
            show_example = False

        output = llm.generate(
            input_prompt,
            sampling_params=SamplingParams(
                n=args.n,
                temperature=args.temperature,
                top_p=0.95,
                max_tokens=args.max_output_tokens,
            ),
            use_tqdm=False
        )

        responses = [t.text for t in output[0].outputs]
        result_data.append({**inference_data, "model_response":responses})

        print(f"Process {process_id} has processed {len(result_data)} data entries...")


    with open(result_save_path, 'w') as f:
        f.write(json.dumps(result_data, indent=4, ensure_ascii=False))
    
    if args.intervention_coeff != 0 and args.intervention_type == "constant":
        close_repcontroller(hooks)
    elif args.intervention_type == "adaptive":
        ada_ctrl.remove_hooks()

    print(f"Process {process_id} finished!")
    return


def make_parser():
    parser = argparse.ArgumentParser()
    # Basic inference parameters
    parser.add_argument("--model_name_or_path", default="", type=str, help="model name or path")
    parser.add_argument("--dataset",  default="", type=str, help="inference dataset name")
    parser.add_argument("--result_save_path", default="", type=str, help="path to save result file")
    parser.add_argument("--max_output_tokens", default=32768, type=int, help="max output tokens")
    parser.add_argument("--max_model_len", default=32768, type=int, help="max model length")
    parser.add_argument("--n", default=8, type=int, help="n")
    parser.add_argument("--temperature", default=0.6, type=float, help="temperature")
    parser.add_argument("--gpu_memory_utilization", default=0.9, type=float, help="gpu memory utilization rate")
    parser.add_argument("--num_gpus", default=1, type=int, help="num gpus")

    # For fast-thinking V.S. slow-thinking experiments
    parser.add_argument("--enable_fast_thinking", action="store_true", help="enable fast thinking")

    # For thinking speed intervention
    parser.add_argument("--intervention_type", default="constant", type=str, help="intervention type, constant or adaptive")

    parser.add_argument("--interventor_ckpt", default="", type=str, help="interventor ckpt")
    parser.add_argument("--intervention_coeff", default=0, type=int, help="intervention coeff")
    parser.add_argument("--intervention_start_layer", default=18, type=int, help="intervention start layer")
    parser.add_argument("--intervention_end_layer", default=26, type=int, help="intervention end layer")

    # For adaptive control
    parser.add_argument("--adaptive_controller_monitor_start_layer", default=0, type=int, help="adaptive controller monitor start layer")
    parser.add_argument("--adaptive_controller_monitor_end_layer", default=14, type=int, help="adaptive controller monitor end layer")
    parser.add_argument("--adaptive_controller_monitor_layer_interval", default=2, type=int, help="adaptive controller monitor layer interval")
    parser.add_argument("--adaptive_controller_min_coeff", default=-4, type=int, help="adaptive controller min coeff")
    parser.add_argument("--adaptive_controller_max_coeff", default=8, type=int, help="adaptive controller max coeff")
    parser.add_argument("--adaptive_controller_sliding_window_size", default=8, type=int, help="adaptive controller sliding window size")
    parser.add_argument("--adaptive_controller_sliding_window_threshold", default=2, type=float, help="adaptive controller sliding window threshold")

    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    print('#'*100)
    print(args)
    print('#'*100)

    # Enable multiprocessing environment
    torch.multiprocessing.set_start_method("spawn")

    # Load input data
    input_data = json.loads(open(f'./datasets/{args.dataset}.json', 'r').read())
    random_idx = np.random.permutation(len(input_data))
    input_data = [input_data[x] for x in random_idx]

    # Create tmp result save paths
    tmp_result_save_folder = f"{args.result_save_path}.tmp"
    os.makedirs(tmp_result_save_folder, exist_ok=True)
    tmp_result_save_paths  = [os.path.join(tmp_result_save_folder, f"Process_{pid}.json") for pid in range(args.num_gpus)]

    inference_processes = []
    input_data_queue = Queue()
    for data in input_data:
        input_data_queue.put(data)
    for pid in range(args.num_gpus):
        process = Process(target=inference, args=(args, input_data_queue, tmp_result_save_paths[pid], pid))
        inference_processes.append(process)
        process.start()
    
    for process in inference_processes:
        process.join()

    output_data = []
    for tmp_result_save_path in tmp_result_save_paths:
        output_data.extend(json.loads(open(tmp_result_save_path, 'r').read()))

    with open(args.result_save_path, 'w') as f:
        f.write(json.dumps(output_data, indent=4, ensure_ascii=False))

    shutil.rmtree(tmp_result_save_folder)

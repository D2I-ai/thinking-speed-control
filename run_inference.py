import os
import json
import shutil
import argparse
from multiprocessing import Process, Queue
import pickle

import torch
import torch.nn.functional as F
import numpy as np
from vllm import LLM, SamplingParams

from utils import PCARepReader



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

    if args.intervention_coeff != 0:
        direction_finder = pickle.load(open(args.interventor_ckpt, "rb"))
        hooks = set_repcontroller(llm.llm_engine.model_executor.driver_worker.model_runner.model, 
                                    list(range(args.intervention_start_layer, args.intervention_end_layer+1,)), 
                                    direction_finder, 
                                    control_coeff=args.intervention_coeff, 
                                    device=f"cuda")


    result_data = []

    show_example = True

    while not data_queue.empty():
        try:
            inference_data = data_queue.get_nowait()
        except Exception as e:
            print("Exception occurred:", e)
            continue
        
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
    
    if args.intervention_coeff != 0:
        close_repcontroller(hooks)

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
    parser.add_argument("--interventor_ckpt", default="", type=str, help="interventor ckpt")
    parser.add_argument("--intervention_coeff", default=0, type=int, help="intervention coeff")
    parser.add_argument("--intervention_start_layer", default=18, type=int, help="intervention start layer")
    parser.add_argument("--intervention_end_layer", default=26, type=int, help="intervention end layer")

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
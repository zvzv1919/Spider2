#!/usr/bin/env python3

import subprocess
import os

def get_idle_gpus(threshold_mb: int = 1000) -> list[int]:
    """Find GPUs with memory usage below threshold (default 1GB)."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    idle_gpus = []
    for line in result.stdout.strip().split('\n'):
        idx, mem_used = line.split(', ')
        if float(mem_used) < threshold_mb:
            idle_gpus.append(int(idx))
    return idle_gpus

def create_offline_llm(model: str, max_model_len: int, gpu_memory_utilization: float, tensor_parallel_size: int):
    gpus = get_idle_gpus()
    if len(gpus) < 2:
        raise RuntimeError("At least 2 GPUs are required to run the model.")
    cuda_devices = ','.join(map(str, gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
    print("Using GPUs: ", cuda_devices)

    from vllm import LLM
    return LLM(
        model=model,
        tensor_parallel_size=len(gpus),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

if __name__ == "__main__":
    from vllm import SamplingParams
    llm = create_offline_llm(
        # model="Qwen/Qwen2.5-7B-Instruct",
        model="Snowflake/Arctic-Text2SQL-R1-7B",
        max_model_len=32000,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )
    
    # Demo
    prompts = ["Hello!", "What is 2+2?"]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=50))
    for out in outputs:
        print(f"\n{out.prompt} → {out.outputs[0].text}")
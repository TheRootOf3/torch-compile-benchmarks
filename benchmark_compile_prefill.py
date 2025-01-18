""" Script for benchmarking the inference with LLMs in PyTorch eager and compiled modes.

This code follow the LLM inference optimisation guide by HF.
https://huggingface.co/docs/transformers/main/en/llm_optims?static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile
"""

from transformers import AutoTokenizer, LlamaForCausalLM, StaticCache
import torch
from typing import Optional, Callable

import time
import os
import csv
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LLAMA3_MODEL_PATH = "/Users/aszab/repos/models/Llama-3.2-1B"
PROMPTS_PATH = "./prompts.txt"


def load_prompts(num_prompts: int) -> list[str]:
    prompts = []
    with open(PROMPTS_PATH, "r") as file:
        for _ in range(num_prompts):
            line = file.readline().strip()
            if line == "":
                print(f"End of file reached. Loaded {len(prompts)} prompts.")
                break

            prompts.append(line)

    return prompts


def prepare_cache(model, inputs):
    batch_size, seq_length = inputs["input_ids"].shape
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=4096,
        dtype=model.dtype,
    )
    cache_position = torch.arange(seq_length)
    return past_key_values, cache_position


def measure_prefill_latency(fn, fn_kwargs):
    # Reset the compiler caches to ensure no reuse between different runs
    torch.compiler.reset()
    with torch._inductor.utils.fresh_inductor_cache():
        start = time.perf_counter()
        with torch.no_grad():
            fn(**fn_kwargs)
        # torch.cuda.synchronize()
        end = time.perf_counter()
        return end - start


def run_experiment(prompts: list[str], compile_fn: Optional[Callable] = None) -> float:

    model = LlamaForCausalLM.from_pretrained(LLAMA3_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA3_MODEL_PATH, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, pad_to_multiple_of=64
    ).to(model.device)

    past_key_values, cache_position = prepare_cache(model, inputs)
    fn_kwargs = {
        **inputs,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "return_dict": False,
        "use_cache": True,
    }

    if compile_fn is not None:
        model = compile_fn(model)

    latency = measure_prefill_latency(model, fn_kwargs)
    del model, tokenizer, inputs, past_key_values, cache_position, fn_kwargs
    gc.collect()

    return latency


def compile_fn(model):
    return torch.compile(model, mode="reduce-overhead")


def main():
    prompts = load_prompts(2)

    latency = run_experiment(prompts, compile_fn)
    print(f"Latency: {latency:.4f} s")

    latency = run_experiment(prompts)
    print(f"Latency: {latency:.4f} s")

    # latency = run_experiment(prompts, None)
    # print(f"Latency: {latency:.4f} s")

    # latency = run_experiment(prompts, None)
    # print(f"Latency: {latency:.4f} s")

    # for layer in model.model.layers:
    #     layer.mlp.forward = torch.compile(layer.mlp.forward, mode="reduce-overhead")
    #     layer.self_attn.forward = torch.compile(
    #         layer.self_attn.forward, mode="reduce-overhead"
    #     )

    # for layer in model.model.layers:
    #     layer.forward = torch.compile(layer.forward, mode="reduce-overhead")

    # model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    # compiled_latency = run_experiment(model, inputs)
    # print(f"Compiled latency: {compiled_latency:.4f} s")


if __name__ == "__main__":
    main()

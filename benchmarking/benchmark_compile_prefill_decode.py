""" Script for benchmarking the inference with LLMs in PyTorch eager and compiled modes.

This code follow the LLM inference optimisation guide by HF.
https://huggingface.co/docs/transformers/main/en/llm_optims?static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile
"""

from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from typing import Optional, Callable

import os
import gc

from utils import (
    load_prompts,
    prepare_cache,
    measure_prefill_latency,
    benchmark_decode_one_token,
    save_experiment_results,
    get_experiment_name,
)
from compile_functions import (
    compile_model_fn,
    compile_layers_fn,
    compile_mlp_attn_fn,
)

device = "cpu"
# device = "cuda:0"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LLAMA3_MODEL_PATH = "/Users/aszab/repos/models/Llama-3.2-1B"


def run_model(
    prompts: list[str],
    num_tokens_to_generate: int,
    compile_fn: Optional[Callable] = None,
    compile_for_prefill: bool = True,
    print_results: bool = False,
    log_decodings: bool = False,
) -> dict[str, float]:

    model = LlamaForCausalLM.from_pretrained(LLAMA3_MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA3_MODEL_PATH, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        # pad_to_multiple_of=(
        #     64 if compile_fn is not None else None
        # ),  # only pad when compiling, otherwise let eager figure out what to do
    ).to(model.device)

    past_key_values, cache_position, generated_ids = prepare_cache(
        model, inputs, num_tokens_to_generate
    )
    fn_kwargs = {
        **inputs,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "return_dict": False,
        "use_cache": True,
    }

    if compile_for_prefill and compile_fn is not None:
        model = compile_fn(model)

    next_token, start_prefill, end_prefill = measure_prefill_latency(model, fn_kwargs)

    _, seq_length = inputs["input_ids"].shape
    generated_ids[:, seq_length] = next_token[:, 0]
    cache_position = torch.tensor([seq_length + 1]).to(model.device)

    if not compile_for_prefill and compile_fn is not None:
        model = compile_fn(model)

    generated_ids, time_list, gen_time_list = benchmark_decode_one_token(
        model,
        num_tokens_to_generate,
        next_token,
        cache_position,
        past_key_values,
        generated_ids,
    )

    # report times
    if print_results:
        print(f"Prefill latency: {end_prefill-start_prefill:.4f} s")
        print(f"Decode latency (first): {time_list[0]-end_prefill:.4f} s")
        print(f"Decode latency (rest): {time_list[-1]-time_list[0]:.4f} s")
        print(
            f"Decode latency per token (rest): {(time_list[-1]-time_list[0])/num_tokens_to_generate:.4f} s"
        )
        print(f"Full latency: {time_list[-1]-start_prefill:.4f} s")

    # form a results dictionary
    results = {
        "start_prefill": start_prefill,
        "end_prefill": end_prefill,
        "time_list": time_list,
        "gen_time_list": gen_time_list,
    }
    if log_decodings:
        results["decodings"] = (
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
        )

    # clean after the experiment
    del model, tokenizer, inputs, past_key_values, cache_position, fn_kwargs
    gc.collect()

    return results


def run_experiment(prompts: list[str], experiment_config: dict) -> float:
    print(f"\n---Run {experiment_config['name']}---")
    # warmup
    for _ in range(experiment_config["num_warmup_steps"]):
        print("Warmup started")
        _ = run_model(
            prompts,
            experiment_config["num_tokens_to_generate_warmup"],
            experiment_config.get("compile_fn", None),
            experiment_config.get("compile_for_prefill", True),
        )
        print("Warmup done")

    # proper run
    results = run_model(
        prompts,
        experiment_config["num_tokens_to_generate"],
        experiment_config.get("compile_fn", None),
        experiment_config.get("compile_for_prefill", True),
        print_results=True,
        log_decodings=True,
    )

    save_experiment_results(
        results,
        experiment_config.copy(),
        filename=f"./results_final/{experiment_config['name']}.json",
    )


def main():

    # for num_prompts in [1, 2, 4, 8, 32, 128]:
    for num_prompts in [128]:
        experiment_config = {
            "num_tokens_to_generate": 200,
            "compile_fn": None,
            "num_prompts": num_prompts,
            "compile_for_prefill": False,
            "num_warmup_steps": 1,
            "num_tokens_to_generate_warmup": 20,
        }
        experiment_config["name"] = get_experiment_name(experiment_config)

        prompts = load_prompts(experiment_config["num_prompts"])

        # run eager
        run_experiment(prompts, experiment_config)

        # compile the model (decode)
        experiment_config["compile_fn"] = compile_model_fn
        experiment_config["name"] = get_experiment_name(experiment_config)
        run_experiment(prompts, experiment_config)

        # compile the model (prefill+decode)
        experiment_config["compile_fn"] = compile_model_fn
        experiment_config["compile_for_prefill"] = True
        experiment_config["name"] = get_experiment_name(experiment_config)
        run_experiment(prompts, experiment_config)


if __name__ == "__main__":
    main()

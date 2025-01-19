from transformers import StaticCache
import torch
import json

import time


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


def prepare_cache(model, inputs, num_tokens_to_generate):
    batch_size, seq_length = inputs["input_ids"].shape
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=4096,
        dtype=model.dtype,
    )
    cache_position = torch.arange(seq_length)
    generated_ids = torch.zeros(
        batch_size, seq_length + num_tokens_to_generate + 1, dtype=torch.int
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(torch.int)
    return past_key_values, cache_position, generated_ids


def measure_prefill_latency(fn, fn_kwargs) -> tuple[torch.Tensor, float, float]:
    # Reset the compiler caches to ensure no reuse between different runs
    torch.compiler.reset()
    with torch._inductor.utils.fresh_inductor_cache():
        start = time.perf_counter()
        with torch.no_grad():
            logits = fn(**fn_kwargs)[0]
            # torch.cuda.synchronize()
            new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        end = time.perf_counter()

    return new_token, start, end


def decode_one_token(
    model, cur_token, cache_position, past_key_values
) -> tuple[torch.Tensor, float, float]:
    # Reset the compiler caches to ensure no reuse between different runs

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(
            cur_token,
            position_ids=None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True,
        )[0]
        new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    end = time.perf_counter()

    return new_token, start, end


def benchmark_decode_one_token(
    model,
    num_tokens: int,
    next_token,
    cache_position,
    past_key_values,
    generated_ids,
):
    torch.compiler.reset()
    with torch._inductor.utils.fresh_inductor_cache():
        time_list = []
        gen_time_list = []
        for _ in range(num_tokens):
            next_token, start_time, end_time = decode_one_token(
                model,
                next_token.clone(),
                cache_position,
                past_key_values,
            )
            time_list.append(end_time)
            gen_time_list.append(end_time - start_time)
            generated_ids[:, cache_position] = next_token.int()
            cache_position += 1

    return generated_ids, time_list, gen_time_list


def save_experiment_results(results, experiment_config, filename):
    experiment_config["compile_fn"] = experiment_config["compile_fn"].__name__

    with open(filename, mode="w") as file:
        json.dump(
            {"experiment_config": experiment_config, "results": results},
            file,
        )

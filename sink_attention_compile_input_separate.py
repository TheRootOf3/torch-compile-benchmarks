# This script allows to do 3 things:
# - Measure how much time it takes for the model to recompile and how many recompilations will occur
# - Test if the order of requests arriving affects the performance.
# - Test how different cache implementation work with torch.compile
#
# Note that all of these experiments compare the performance of torch.compile with the performance of the model without torch.compile.
# Furthermore, we use two distinct instances of a model for 1) the prefill stage and 2) decoding.


import os
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    SinkCache,
    StaticCache,
    DynamicCache,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

text = ""
with open("./cupcake_ipsum.txt", "r") as f:
    for line in f.readlines():
        text += line.rstrip()

assert len(text) > 0

LLAMA_3_PATH = "/Users/aszab/repos/models/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(LLAMA_3_PATH)
model_prefill = AutoModelForCausalLM.from_pretrained(LLAMA_3_PATH)
model_decode = AutoModelForCausalLM.from_pretrained(LLAMA_3_PATH)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model_prefill.forward = torch.compile(
    model_prefill.forward, mode="reduce-overhead", dynamic=True
)
model_decode.forward = torch.compile(
    model_decode.forward, mode="reduce-overhead", dynamic=True
)

size_list = [5, 10, 20, 50, 100, 250, 500, 1000, 1500, 5000][::-1]
generate_list = [10, 30, 100, 300]
for i in size_list:
    gen_times = []
    for j in generate_list:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=i,
            # pad_to_multiple_of=64,
            # padding_side="left",
        ).to(model_prefill.device)

        assert inputs["input_ids"].shape[1] == i
        # past_key_values = SinkCache(window_length=6000, num_sink_tokens=4)
        past_key_values = StaticCache(
            model_prefill.config, max_batch_size=1, max_cache_len=6000
        )
        decoding = []
        start_time = time.time()
        with torch.no_grad():
            out = model_prefill(
                **inputs,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values

            last_token = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            decoding.append(last_token.squeeze())

            for num_to_generate in range(j - 1):
                out = model_decode(
                    last_token,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                last_token = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                decoding.append(last_token.squeeze())

        gen_times.append(time.time() - start_time)

        assert len(decoding) == j
        # print(tokenizer.batch_decode([decoding], skip_special_tokens=True)[0])

    print(i, list(zip(generate_list, gen_times)))

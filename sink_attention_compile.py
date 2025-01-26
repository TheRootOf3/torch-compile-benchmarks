import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LLAMA_3_PATH = "/Users/aszab/repos/models/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(LLAMA_3_PATH)
model = AutoModelForCausalLM.from_pretrained(LLAMA_3_PATH)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(
    "This is a long story about unicorns, fairies and magic.", return_tensors="pt"
).to(model.device)

# get our cache, specify number of sink tokens and window size
# Note that window size already includes sink tokens, so has to be larger
past_key_values = SinkCache(window_length=256, num_sink_tokens=4)

model.forward = torch.compile(model.forward)

out = model.generate(
    **inputs, do_sample=False, max_new_tokens=30, past_key_values=past_key_values
)
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])

past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
out = model.generate(
    **inputs, do_sample=False, max_new_tokens=100, past_key_values=past_key_values
)
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])

past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
out = model.generate(
    **inputs, do_sample=False, max_new_tokens=300, past_key_values=past_key_values
)
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])


past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
out = model.generate(
    **inputs, do_sample=False, max_new_tokens=320, past_key_values=past_key_values
)
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])

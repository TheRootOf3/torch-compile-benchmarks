"""Compare the performance of a model with and without torch.compile.

Run in the following way for torch compilation logs:
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCH_COMPILE_DEBUG=1 python3 compare_llm.py
"""

import torch

# from transformers import GPT2LMHeadModel
from model import GPT, GPTConfig, MLP, CausalSelfAttention

GPT2_MODEL_PATH = "/Users/aszab/repos/models/gpt2"

config = GPTConfig()
# m1 = MLP(config)
m1 = CausalSelfAttention(config)

# m1 = GPT(GPTConfig())
# m1 = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH)

seq_length = 128
bs = 32
# vocab_size = m1.config.vocab_size
# input = torch.randint(0, vocab_size, (bs, seq_length), dtype=torch.int64)
input = torch.rand((bs, seq_length, config.n_embd))

# Initialize the inductor model
compiled_model = torch.compile(m1)
with torch.no_grad():
    compiled_model(input)

NUM_ITERS = 50
import timeit

with torch.no_grad():
    # warmup
    for _ in range(10):
        m1(input)
    eager_t = timeit.timeit("m1(input)", number=NUM_ITERS, globals=globals())

with torch.no_grad():
    # warmup
    for _ in range(10):
        compiled_model(input)
    inductor_t = timeit.timeit(
        "compiled_model(input)", number=NUM_ITERS, globals=globals()
    )
print(f"eager use: {eager_t * 1000 / NUM_ITERS} ms/iter")
print(f"inductor use: {inductor_t * 1000 / NUM_ITERS} ms/iter")
print(f"speed up ratio: {eager_t / inductor_t}")

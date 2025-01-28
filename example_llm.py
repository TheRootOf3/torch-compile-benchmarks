"""Example of a simple script that loads a model and JIT compiles it.

Run in the following way for torch compilation logs:
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCH_COMPILE_DEBUG=1 python3 example_llm.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLAMA3_MODEL_PATH = "/Users/aszab/repos/models/Llama-3.2-1B"
GPT2_MODEL_PATH = "/Users/aszab/repos/models/gpt2"

model = AutoModelForCausalLM.from_pretrained(LLAMA3_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(LLAMA3_MODEL_PATH)


compiled_model = torch.compile(model)

prompt = "Hello, my dog is cute and"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = compiled_model(
        **inputs,
    )

from model import GPT, GPTConfig
import torch

from torch._inductor import config

config.cpp.enable_kernel_profile = True


# bench.py
from torch.profiler import profile, schedule, ProfilerActivity

RESULT_DIR = "./prof_trace"
my_schedule = schedule(skip_first=10, wait=5, warmup=5, active=1, repeat=5)


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
    print(output)
    p.export_chrome_trace(f"{RESULT_DIR}/{p.step_num}.json")


model = GPT(GPTConfig())
compiled_model = torch.compile(model, backend="inductor")
seq_length = 128
bs = 4
vocab_size = model.config.vocab_size
input = torch.randint(0, vocab_size, (bs, seq_length), dtype=torch.int64)

for _ in range(10):
    # model(input)  # compiled_model(**input_dict) to get inductor model profiling
    compiled_model(input)

total = 0
with profile(
    activities=[ProfilerActivity.CPU],
    schedule=my_schedule,
    on_trace_ready=trace_handler,
) as p:
    for _ in range(50):
        compiled_model(
            input
        )  # compiled_model(**input_dict) to get inductor model profiling
        p.step()

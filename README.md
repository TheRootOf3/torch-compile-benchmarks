# Studying the Overhead of `torch.compile()` in Large Language Model Inference

📚 This repository contains files used in the project *Studying the Overhead of `torch.compile()` in Large Language Model Inference*.

The structure of the repo is as follows:
- `benchmarking/` -- this directory contains code used for prefill+decode experiments.
- `compare_llm.py` -- profiling comparison between compiled and not compiled models.
- `compile_multiple_inputs.py` -- code for generating dynamically changing prefill sequence lengths.
- `example_llm.py` -- example compiled llm.
- `profile_ops.py` -- script for profiling operations of a PyTorch code (works with both compiled and eager modes). 
- `model.py` -- GPT-2 implementation from https://github.com/karpathy/nanoGPT.
- `visualise_results.ipynb` -- notebook containing code for plotting results in the report.
- `prompts2.txt` -- file containing synthetic prompts for experiments.
- `cupcake_ipsum.txt` -- file containing synthetically long context.
- `benchmarking_results_final/` -- results from prefill+decode experiments.
- `resources/` -- resources for the report, e.g. images, plots, etc.
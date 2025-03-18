# Studying the Overhead of `torch.compile()` in Large Language Model Inference

The report is available [here](/report.pdf).

> **TL;DR** To lower the latency of initial LLM generations on spot instances, do not compile for the prefill phase. Furthermore, set `torch.compile(dynamic=True)` if expecting different prefill shapes (prompt lengths) to avoid extra recompilations. It can be also approached by padding on the left until a fixed sequence length (but this has the cost of fixed prefill shape and may lead to suboptimal tensor optimisations).

## Structure of this Repo


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
- `report.pdf` -- a report with some cool insights. Heads up -- it was written overnight to make it for the deadline, so if you spot any typos this is likely the reason 😉  

## Note

This works has been submitted as part of the module assignment for R244: Large Scale Data Optimisation and Processing.
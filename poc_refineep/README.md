# RefineEP refinement-aware third-path PoC

This workspace evaluates whether the compacted fresh-work shapes produced by
LLaDA2.0-Flash refinement leave enough communication/runtime headroom to
justify a new fixed-contract EP4 dispatch/combine path.

The study is oracle-first.  A CUDA prototype is allowed only when the credible
request-level oracle reaches 12% on both GSM8K and HumanEval.

## Fixed substrate

- Physical GPUs: 4,5,6,7 only
- Model: `inclusionAI/LLaDA2.0-flash`
- Dense TP4, routed EP4, 256 experts, top-k 8, hidden 4096, BF16
- Existing runtime: DeepEP intranode normal path; valid low-latency path is
  measured as a second existing baseline

Raw measurement directories under `results/` are intentionally ignored.  The
machine-readable summaries and reports are tracked.

## Final result

`NO-KERNEL-HEADROOM`.  Exact liveness filtering produces a broad M=1--1024
refinement continuum, but existing DeepEP low-latency wins every real and
controlled route.  The credible third-path increment is 2.29%/2.10% request
E2E on GSM8K/HumanEval, below both the 5% kill gate and 12% CUDA gate.  No new
kernel was implemented.

# Track B6 — live approximate PP decision

No live approximate PP speedup is claimed.

Stage-local PP2xEP2 and PP4 weights load successfully, and process groups plus
activation transport are valid.  However, the current dInfer LLaDA2 diffusion
runner lacks the inter-stage forward/KV contract needed for an exact sequential
PP reference.  A trustworthy live asynchronous implementation would require a
non-trivial PP-aware model return path, diffusion-state broadcast, stage-local
KV cache, and quality validation.

That engineering is not justified by the gate:

- no common quality-safe analytical point reaches 1.2x;
- the best common point is only 1.184x before real PP overhead;
- current upstream LLaDA2 serving paths also do not support PP;
- direct prior art already covers stale diffusion pipelines.

Result: **live prototype withheld by both environment and headroom gates**.
This is not reported as method failure or as a measured 1.184x speedup.

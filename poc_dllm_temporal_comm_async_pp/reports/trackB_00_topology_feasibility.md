# Track B0 — topology feasibility

## What is actually supported

SGLang 0.5.3 can construct both target process-group geometries:

- PP2xEP2: stage 0 ranks `[0,1]`, stage 1 ranks `[2,3]`; PP peers
  `[0,2]` and `[1,3]`; EP groups `[0,1]` and `[2,3]`.
- PP4: one rank/stage with PP group `[0,1,2,3]`; no within-stage EP.

After a bounded dInfer loader fix to skip non-local `PPMissingLayer`
placeholders, stage-local weights also fit without CPU offload:

| Topology | Layer placement | Routed ownership | Allocated HBM/rank |
|---|---|---|---:|
| PP2xEP2 | 0--15 / 16--31 | 128 experts/rank within each stage | 51.1--52.9 GiB |
| PP4 | 0--7 / 8--15 / 16--23 / 24--31 | all 256 experts on stage rank | 47.7--52.4 GiB |

Neighboring-rank BF16 activation transport is inexpensive relative to a stage:
0.040--0.079 ms p50 for 0.25--8 MiB, versus measured quarter-stage medians of
roughly 28--34 ms.

## Remaining blocker

This is not yet an exact request-level PP runtime.  dInfer's LLaDA2 wrapper
unconditionally expects a final-stage `(hidden, present)` result and executes
the LM head, while its block-diffusion KV manager stacks all 32 layers on every
rank.  These contracts do not implement inter-stage diffusion-state transport
or stage-local KV ownership.  The current upstream
[vLLM dLLM plugin](https://github.com/vllm-project/dllm-plugin/blob/main/docs/OPERATOR_LLaDA2.md)
also explicitly lists PP>1 as unsupported for LLaDA2.

Therefore:

- **topology/group construction: supported**;
- **stage-local parameter capacity: supported**;
- **exact sequential PP request execution: environment/port blocked**;
- **live asynchronous PP: not claimed**.

Evidence: `trackB/topology/*.json`, `trackB/*_load_audit/*.json`, and logs in
the result root.

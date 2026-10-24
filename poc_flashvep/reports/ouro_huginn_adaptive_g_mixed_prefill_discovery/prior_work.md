# Prior work and claim boundaries

## Publication, implementation, and candidate

| Evidence | Established feature | This study's boundary |
| --- | --- | --- |
| [CDB paper, v2](https://arxiv.org/abs/2608.09444v2) | Recurrent batching and dynamic boundary/core execution | Per-loop scheduling is not new |
| [Official CDB code](https://github.com/kschwethelm/continuous-depth-batching) | Ouro/Huginn runtime and looped KV management | This study does not replace its model gate or train a gate |
| [Pinned RLT](https://github.com/InferMatrixLab/vllm-rlt/tree/4a350344027960a48384626aa646f1765fda6d4f) | Atomic, Split, and Mixed execution paths | Mixed is an established strong baseline |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked prefill and decode-maximal hybrid batching | Prefill/decode co-batching is not a contribution |
| [Layered Prefill](https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html) | Physical-layer groups, motivated by MoE weight traffic | Dense recurrent grouping is a different axis, not automatic novelty |
| [Threshold-Based Exclusive Batching](https://proceedings.mlr.press/v306/zhang26eo.html) | State-dependent mixed/exclusive execution | A generic conditional switch is not new |
| [Huginn model card](https://huggingface.co/tomg-group-umd/huginn-0125/tree/bb6621b65e90b6a4b9b29ef88dc83866d450470c) | Fixed injection, recurrent KV, and configurable inference depth | Preserve model semantics and supported context |

The CDB abstract reports strong gains for its own tested architecture and exit behavior.
That claim does not prove this fixed-depth prefill hybrid has zero headroom.
The source audit, not an abstract alone, determines the implemented controls.
The local CDB checkout resolves to `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
Its `scheduler_base.py` explicitly describes phase-separated prefill and decode batches.
That source statement does not imply that every CDB version or implementation excludes Mixed.
The RLT pin already exposes loop count in its execution descriptor.
It does not provide the tested online multi-G prompt/Mixed selector as a stock policy.
This statement concerns the pinned implementation, not every version of CDB or every recurrent scheduling paper.

## Preserved negative evidence

The earlier fixed G2 and SLO study found a TTFT–tail trade-off.
Its unmatched-C serving points do not prove superiority over tuned Mixed.
Compute-budget Joint selected the same 144/144 executed plans as adaptive G-only.
Dual-objective Joint changed C in 61/532 fresh decisions without a matched serving win.
Decode Elastic Loop Batching failed incremental promotion against native async/Mixed.
These findings remain unchanged.

This experiment changes the question from prefill C/G selection or decode grouping.
C remains fixed. Only legal prompt grouping and native Mixed composition can change.
The required new evidence is an incremental complete-request operating region beyond a strong fixed hybrid.
No candidate receives novelty credit merely because it selects an existing execution path.
Exclusive-batching thresholds from other models or GPUs do not define a valid H100 threshold here.

## Additional current-source checks

The fresh search used recurrent prefill grouping and mixed-batching terms.
The following primary sources define nearby but different control axes.

| Primary source | Control axis | Scope here |
| --- | --- | --- |
| [SLOWeave](https://arxiv.org/abs/2609.07883) | Deadline-aware token chunk width | C remains fixed here. Its predictor assumptions do not become a guarantee |
| [P-PAS v2](https://arxiv.org/abs/2608.15171v2) | Global token budget from prefill/decode pressure | A pressure switch alone is not recurrent-group novelty |
| [TAPS](https://arxiv.org/abs/2609.36653) | Recurrent update scale | It changes the mathematical update, not just this execution schedule |
| [RAMP](https://arxiv.org/abs/2610.13041) | Recurrent mixed precision | Precision reduction is outside this unchanged-BF16 contract |
| [DLFP](https://arxiv.org/abs/2609.38386) | Decode-feedback token chunking | Its reported proxy-timing limits support separating host intervals from complete GPU work |

These checks do not constitute an exhaustive novelty search.
No evidence in this campaign establishes a new scheduling contribution beyond the listed controls.

# Official Layered Prefill upstream audit

- Repository: `https://github.com/scale-snu/layered-prefill`
- Inspected checkout: `053f80e5201a7c0ab56e468e3d578907a2ca9cc3`
- Audit date: 2026-09-29
- Files inspected in full: `README.md`, `nanovllm/engine/scheduler.py`, and `nanovllm/engine/model_runner.py`.

The public implementation is a nano-vLLM implementation whose distributed model runner initializes a tensor-parallel world. It does not expose Qwen3-VL, expert-parallel ownership, or DeepEP dispatch/combine support. It therefore cannot be reported as a native EP baseline.

The scheduling semantics retained by this PoC are:

- a prefill advances through contiguous layer groups;
- all scheduled prefill tokens remain present at an active layer group;
- intermediate hidden and residual state is retained between groups;
- decode is scheduled on every iteration;
- the method does not token-chunk, drop tokens, replace experts, or change routing.

The authoritative spec says that 2,029 tokens should select eight stages. The pinned upstream code does **not** do that: its exact comparisons select four stages for `num_batched_tokens <= 2048`, so 2,029 selects four stages. Eight stages begin at 2,049 tokens. Resident B8 decode adds at most eight scheduled tokens and leaves the request at 2,037, still in the four-stage branch. The replay follows the inspected implementation: four contiguous groups of twelve layers. This discrepancy is reported rather than hidden.

A native adapter would require resumable layer-group execution plus KV/intermediate-state ownership across the installed vLLM external-DP/TP2/EP4 worker contract. That is beyond a bounded motivation study and risks creating a new runtime rather than evaluating the baseline. RQ3 consequently uses `LAYERED_EP_PHYSICAL_REPLAY`: every active-layer communication datum comes from an actually executed TRUE-EP4 DeepEP full-prompt collective, and the four-stage grouping is composed offline. It is not described as native Layered serving, and no native TTFT/E2E claim is made.

# EP Simulation Validation

Date: 2026-06-25

## Goal

Validate whether fast simulation can use a single-GPU forward's routing plus a
static expert-to-rank mapping to reproduce the actual vLLM 8-way EP routing
loads. The requested mapping was the vanilla linear placement:

```text
rank = expert_id // 16
0-15 -> rank0, 16-31 -> rank1, ..., 112-127 -> rank7
```

## Inputs And Commands

The comparison used the same dummy multimodal prompt as
`scripts/vllm_ep_sanity.py`: one 224x224 gray image plus the text prompt
`Describe this image briefly.` Sampling used `max_tokens=1`, so the returned
routing is prefill-only for 79 prompt tokens.

8-way EP collection:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 scripts/vllm_collect_routing.py collect \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --output-prefix outputs/ep_sim_validation/ep_routing
```

Single-GPU collection:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 scripts/vllm_collect_routing.py collect \
  --tensor-parallel-size 1 \
  --output-prefix outputs/ep_sim_validation/single_routing
```

Comparison:

```bash
python3 scripts/vllm_collect_routing.py compare \
  --ep-npz outputs/ep_sim_validation/ep_routing.npz \
  --single-npz outputs/ep_sim_validation/single_routing.npz \
  --output outputs/ep_sim_validation/compare.json
```

## Results

Both paths returned routed-expert tensors with the same shape:

- EP: `[79, 48, 8]`
- single GPU: `[79, 48, 8]`
- total routing entries: `30,336`

Memory confirmed the difference between actual EP and full replication:

- 8-way EP: `11,825 MiB` per GPU after load
- single GPU: `61,565 MiB` on GPU0 after load

The exact routing did not match:

- exact routing match: `false`
- entry mismatches: `16,224 / 30,336`
- expert-load equality: `false`
- expert-load L1 difference: `898`
- rank-load equality: `false`
- rank-load L1 difference: `106`

Rank loads:

| rank | EP load | single+mapping load |
| ---: | ---: | ---: |
| 0 | 3650 | 3653 |
| 1 | 3635 | 3634 |
| 2 | 4085 | 4082 |
| 3 | 3660 | 3657 |
| 4 | 4023 | 4073 |
| 5 | 3582 | 3575 |
| 6 | 4019 | 4017 |
| 7 | 3682 | 3645 |

Top hot experts were highly similar but not identical in order:

- EP top experts: `98, 107, 45, 8, 65, 40, 26, 38, 111, 57`
- single top experts: `98, 107, 8, 45, 65, 40, 38, 26, 111, 57`

## Interpretation

The user-requested validation gate does not pass as an exact equivalence:
single-GPU routing plus linear expert-to-rank mapping is not bit-for-bit
identical to actual vLLM 8-way EP routing.

The rank-level load difference is small for this dummy input
(`106 / 30,336 = 0.35%` L1 over routing entries), so the single-GPU path may be
a useful approximation. However, it is not rigorous enough to justify all later
measurement from single-GPU simulation without an explicit tolerance policy.

Likely cause: the comparison changes both physical EP and the tensor-parallel
arithmetic path. Actual EP was run as `tensor_parallel_size=8` with MoE EP over
the 8 ranks; the single-GPU run used `tensor_parallel_size=1`. Small numerical
differences before the router can change top-k membership for close experts.

I started a TP=8 no-EP control run to isolate "TP arithmetic" from "EP sharding",
but stopped it after the requested EP-vs-single gate had already failed and the
control run was still loading after several minutes. No Phase 2-A motivation or
calibration measurements were run after this failed gate.

## Status

Gate result: **failed for exact simulation equivalence**.

Recommended next step before resuming Phase 2-A measurement:

- either define an acceptable load-level tolerance and rerun validation on a
  representative sample set,
- or use actual vLLM EP routed-experts capture for Phase 2-A load/motivation
  profiling instead of relying on single-GPU simulation.

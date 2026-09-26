# Virtual EP architecture

The implementation separates semantic evidence, workload replay, and latency
prediction. A target EP number never changes `CUDA_VISIBLE_DEVICES` and a
serial replay is never reported as parallel EP wall time.

## A. Semantic runner

`scripts/run_full_model.py` executes the full Qwen vision encoder, language
model, router, sparse experts, and generation. It supports:

- stock reference TP4 and physical TP2/DP2/EP4 on GPUs 4–7;
- stock reference TP2 on GPUs 4,5;
- stock reference TP1 in a single-process vLLM mode on GPU 4.

Only actual recomputed generation may support a quality statement. Trace
editing cannot infer downstream routing or quality for a new resolution/top-k.

## B. Workload replay

`virtual_ep/replay.py` consumes real top-8 routing arrays and constructs a
linear 128-expert owner map for target EP4 or EP8. It emits per-expert and
per-rank histograms, source-destination deduplicated rows, logical dispatch and
combine bytes, active experts, fanout, and evidence labels.

The frozen 24-request replay covers all 48 routed layers:

| Field | Virtual EP4 | Virtual EP8 |
|---|---:|---:|
| Layer/request rows | 1,152 | 1,152 |
| Assignments | 3,016,704 | 3,016,704 |
| Logical rows | 377,088 | 377,088 |
| Remote unique rows | 1,030,241 | 1,763,303 |
| Dispatch logical bytes | 4,219,867,136 | 7,222,489,088 |
| Combine logical bytes | 9,272,221,696 | 10,820,943,872 |
| Maximum source fanout | 3 | 7 |

EP4 is a replay of a measured EP4 topology. EP8 is labeled
`UNVALIDATED_EP8_EXTRAPOLATION`; neither artifact contains a latency claim.

## C. Cost simulator

`virtual_ep/cost.py` validates model/backend/topology signatures, rejects
2048×512 and 4096×1024 stale calibrations, selects measured real-shape
neighbors, and marks shape distance. It emits no confident milliseconds when
calibration is absent or out of domain.

The current nearest-shape simulator failed its held-out accuracy gates. This
does not invalidate the replay layer, but it prevents virtual EP4/8 latency
screening until new calibration is collected.

## CLI contract

`poc_mllm_ep_foundation.cli` implements separate `--physical-gpus` and
`--target-ep` arguments for audit, replay, calibration, validation, and resume.
GPU-producing baseline/anchor commands deliberately stop at a manifest and
delegate to the explicit full-model runner rather than fabricate completion.

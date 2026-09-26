# Calibration shapes and backends

Status: **CALIBRATED_SMALL_COHORT; insufficient for screening**

## Signature

| Field | Value |
|---|---|
| Revision | `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| Hidden / intermediate | 2048 / 768 |
| Experts / top-k | 128 / 8 |
| Backend | DeepEP high-throughput |
| Topology | TP2/DP2/SP1/PP1/EP4 |
| Expert compute | sparse Triton experts |

Calibration caches are backend- and topology-specific. NCCL, TP-only, EP8,
and historical LLaDA2 shapes are not silently reused.

## Measured cohort

The matched profiling pass produced 528 layer-level shape observations from
11 requests. A request-level split used seven calibration requests and four
held-out requests, producing 192 held-out layer observations. This is below
the required 16 calibration + 8 held-out request minimum.

Calibration requests: `diverse2_small`, `diverse4_small`, `diverse8_small`,
`repeat2_coins`, `repeat4_coins`, `repeat8_coins`, `single_coins`.

Held-out requests: `single_grass`, `text_control_large`,
`text_control_medium`, `text_control_small`.

## Component semantics

Each observation records total assignments, critical-rank assignments,
remote unique rows, active experts, and measured dispatch/expert/combine/routed
CUDA spans. DeepEP fused/internal work is not divided using an invented
bandwidth or a 2:1 gate/up/down time split.

Artifacts: `artifacts/analysis/cost_calibration.json` and
`artifacts/analysis/heldout_replay.json`.

# Decode tail geometry diagnostic

Status: two matched clean geometry controls passed. Independent joint confirmation remains pending.

## Reproduction

Evidence class: MEASURED.

Huginn uses original BF16 weights and fixed R32. Each run offers 512 requests with 2048 input tokens and 16 output tokens.
Both controls use Poisson seed 2026111603 and nominal QPS 1.971. Native NIXL transport and the probe policy remain unchanged.
Each run completes twelve D-local, 16-token, full-R32 Prefill probes. These probes do not migrate serving work from P to D.

| D slots | Decode graph row limit | Completed requests | Requests/s | TPOT p95 ms | ITL p99 ms | Maximum ITL ms | Eager fallbacks | Whole-run Decode gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 4 | 512/512 | 1.649274208 | 144.661527773 | 272.710584160 | 775.795314461 | 547 | FAIL |
| 4 | 4 | 512/512 | 1.658745516 | 128.328480373 | 165.353531521 | 456.572681665 | 0 | PASS |
| 8 | 8 | 512/512 | 1.652125953 | 130.198055351 | 189.975844305 | 1203.319249675 | 0 | PASS |

The frozen ITL p99 limit is 192.821253818 ms. The TPOT margin gate is 167.914355323 ms.
The maximum interval exceeds the percentile limit in all runs. A percentile gate does not guarantee a maximum-ITL bound.
The measured request rates are finite-cohort rates. They are not sustainable capacity estimates.

## Correctness

All runs match all 512 background output sequences against the qualified no-probe native reference.
All twelve probes match their qualified inputs, outputs, and fixed depth.
Each run records 33,806,336 recurrent token rows, equal to the expected complete work.
Both changed geometries also pass separate eight-request prompt-KV and hidden-state hash comparisons.
These checks do not establish original-HF fidelity or downstream task accuracy.

## Interpretation

The earlier clean failure includes five-row and six-row Decode submissions beyond the four-row graph limit.
The changed slot limit prevents this oversized batch condition in the measured run.
The observed tail reduction supports the graph-coverage hypothesis. It does not isolate graph coverage from changed batch composition.
The eight-slot, eight-row graph control also removes eager fallback and passes the whole-run percentile gate.
Its ITL p99 margin is only 2.845409513 ms. Independent repeats must test this narrow margin.
The 1203.319250 ms maximum remains unexplained by eager fallback. The change does not remove every tail outlier.

The four-slot result does not yet establish same-run P saturation and D headroom.
Fewer slots can increase D admission waiting even when KV memory remains free.
The next gate requires dual-device traces, unchanged admission criteria, and independent repeats.
No method-performance expansion is authorized by this single clean result.

## Evidence

- `state/tail_geometry_control.json`
- `raw/recovery_huginn_native8_graph4_tail512/`
- `raw/recovery_huginn_slots4_graph4_state8/`
- `raw/recovery_huginn_slots4_graph4_tail512/`
- `raw/recovery_huginn_slots8_graph8_state8/`
- `raw/recovery_huginn_slots8_graph8_tail512/`
- Reproduction: `python -m poc_flashvep.fluidtoken_loop_split.tail_geometry_control`
- Joint confirmation: `python -m poc_flashvep.fluidtoken_loop_split.geometry_confirmation`

The generic unittest discovery command failed because this source directory is a namespace package.
Explicit module invocations remain the supported CPU test command. This discovery failure is not a GPU execution failure.

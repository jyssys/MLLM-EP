# E2E latency mass atlas (fresh live measurements)

## Scope and accounting

Fresh traces used the real Qwen3-VL V1 serving driver, BF16, TP2/DP2/EP4,
DeepEP high-throughput, DBO off, eager mode, and `CUDA_VISIBLE_DEVICES=1,2,3,4`.
Rank rows are collapsed by `(DP, local invocation, layer, phase)` with a
maximum same-DP CUDA span; rank rows are never summed. Warmup/profile rows are
excluded when `active_wave` is available. The stage sum is a MoE-stage mass,
not a request-level E2E claim, because the stock driver only supplies wave
labels rather than exact request-to-layer joins.

## Request-level observations

| Trace | Workload | Requests | E2E p50 / p90 / p99 (ms) |
|---|---|---:|---:|
| `live_mixed_c2` | varied multimodal, concurrency 2 | 48 | 285.6 / 390.4 / 1148.9 |
| `live_text_c8` | fixed text, concurrency 8 | 160 | 656.0 / 1030.1 / 1943.7 |
| `live_vision_hi_c8` | fixed high-resolution image, concurrency 8 | 160 | 757.8 / 1644.8 / 7617.8 |
| `live_mixed_c8` | varied multimodal, concurrency 8 | 224 | 900.9 / 1115.5 / 6235.0 |
| `live_mixed_c8_mbt4096` | same varied templates, MBT=4096 | 84 | 1277.6 / 1865.1 / 2298.4 |
| `live_text_c16` | fixed text, concurrency 16 | 256 | 773.1 / 1759.1 / 2193.5 |
| `live_high_c16` | varied/high volume, concurrency 16 | 144 | 790.9 / 1048.4 / 4127.1 |
| `live_long_text_c8` | long prefill text, concurrency 8 | 64 | 965.3 / 1583.4 / 1659.7 |

The large request spread is real, but it is not attributable to MoE alone:
normal layer-local `T_MoE` is about 1.15–1.40 ms while image preprocessing,
queueing and 48-layer execution contribute to request time.

The MBT control is a genuine live anomaly: at the same M bins, lowering
`max_num_batched_tokens` from 8192 to 4096 raises M=114 layer-local T_MoE
1.155→2.133 ms (dispatch 0.126→0.834 ms; event wait 0.018→0.273 ms) and
request p50 900.9→1277.6 ms. It is classified as **TRIVIAL_ENGINEERING**:
the static token-budget knob reproduces the effect and no orthogonal variable
or non-trivial feasible oracle remains.

## Normal MoE phase mass

| Trace/phase | T_MoE p50 (ms) | Dispatch share | Expert share | Combine share | Event-wait share |
|---|---:|---:|---:|---:|---:|
| text c8 / prefill | 1.168 | 30.4% | 31.0% | 3.4% | 3.3% |
| vision-hi c8 / prefill | 1.218 | 15.1% | 42.2% | 2.5% | 7.9% |
| text c16 / prefill | 1.399 | 28.4% | 34.9% | 2.6% | 14.6% |
| mixed c8 / prefill | 1.164 | 17.8% | 38.4% | 2.3% | 6.0% |
| mixed c8 / decode | 1.151 | 33.3% | 28.5% | 3.5% | 8.4% |
| mixed c2 / prefill | 1.209 | 29.3% | 33.1% | 3.3% | 12.1% |

Extreme maxima are dispatch-dominant (up to 2.64 s in fresh traces), but
their request-level removable mass is not identified by this wave-only hook.
The previous exact request join capped fixed-shape tail removal at 1.09% of
request latency, so this branch does not promote that closed direction.

## Interpretation

The most repeatable new signal is a **concurrency/phase regime change**:
fixed text c8→c16 raises layer-local `T_MoE` p50 by 19.8% and request E2E p50
by 17.8%, while the expert/dispatch/wait mix shifts. Additional c2/c4/c8/c16
runs and matched shape/warmup controls show substantial state variance; this is
an operational batching/throughput trade-off, not a paper direction. A single
text→vision transition reached 2.4 ms T_MoE versus 1.2 ms after matching
warmup, but the independent telemetry-tagged replication was 1.2 ms. The
transition is therefore retained as a state/DVFS diagnostic and not as a
causal oracle.

## Final transition control summary

| control | T_MoE p50 (ms) | event-wait p50 (ms) | interpretation |
|---|---:|---:|---|
| text fixed after text warmup | 1.16–1.18 | 0.018–0.025 | stable baseline |
| text → high-res vision | 2.395 | 0.334 | one high state-transition run |
| text ↔ vision alternation | ~2.12 | ~0.49 | same transition state |
| high-res vision after vision warmup | 1.204 | 0.028 | shape-matched preconditioning |
| text-shape alternation | 1.21–1.25 | 0.02–0.04 | no generic shape alternation penalty |
| vision-shape alternation | ~1.16 | ~0.029 | no vision-only alternation penalty |
| telemetry-tagged text → vision replication | 1.20 | 0.04 | effect absent; clocks ramp 345→1980 MHz |

The direct request-level fixed-tail result remains the binding headroom bound:
1.09% removable request mass in the exact joined historical trace.

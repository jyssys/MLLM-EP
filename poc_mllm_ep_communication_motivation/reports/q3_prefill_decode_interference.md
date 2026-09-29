# Q3 — Prefill/decode interference

Resident decode is global B8 fixed64. A prefill is injected after 16 generated tokens. Clean TPOT uses homogeneous condition waves and the maximum DP caller wall per engine iteration. Trace runs are separate and describe the shared DeepEP collective experienced by the resident decode; they do not invent a row-wise latency split.

| Condition | Mean ITL | Median ITL | P90 ITL | Maximum stall | Paired mean inflation |
|---|---:|---:|---:|---:|---:|
| Decode only | 102.4980 ms | 99.0219 ms | 104.0925 ms | 287.3716 ms | — |
| Ordinary text prefill | 102.9432 ms | 99.0181 ms | 108.1528 ms | 305.8197 ms | 0.434% |
| Large visual prefill | 101.3075 ms | 99.1258 ms | 107.7128 ms | 293.3087 ms | -1.162% |
| Token-matched text prefill | 102.7192 ms | 99.1174 ms | 107.6087 ms | 283.0303 ms | 0.216% |

## Actual mixed-iteration MoE stage

| Condition | Dispatch mean/layer | Expert mean/layer | Combine mean/layer | MoE mean/layer | Communication inflation | Expert inflation |
|---|---:|---:|---:|---:|---:|---:|
| Decode-only matched reference | 0.4156 ms | 0.4762 ms | 0.1634 ms | 1.0552 ms | — | — |
| Ordinary text prefill | 0.6809 ms | 0.5991 ms | 0.2142 ms | 1.4943 ms | 54.604% | 25.815% |
| Large visual prefill | 4.8115 ms | 0.5529 ms | 0.1078 ms | 5.4722 ms | 749.340% | 18.455% |
| Token-matched text prefill | 4.0174 ms | 0.5503 ms | 0.1120 ms | 4.6796 ms | 613.210% | 15.554% |

Inflations use the decode-only trace at each condition's physically observed mixed iteration (18 for text controls, 19 for visual), not an assumed caller-clock alignment.

## Injected prefill traffic

| Condition | Prompt tokens/layer | Routed pairs (48 layers) | Logical dispatch bytes | Peak-rank dispatch bytes |
|---|---:|---:|---:|---:|
| Ordinary text | 22 | 8448 | 15962112 | 4079616 |
| Large visual | 2029 (1980 visual) | 779136 | 1445330944 | 368009216 |
| Token-matched text | 2033 | 780672 | 1450332160 | 370384896 |

Verdict: **VISUAL_PREFILL_EP_COMM_INTERFERENCE_SUPPORTED**.

The single injected burst is diluted in the 63-step aggregate TPOT. The primary interruption statistic is therefore the common five-iteration response-window maximum (caller iterations 16–20, covering the measured external-DP async lag): ordinary text 4.887% (95% CI -0.526%, 14.782%), token-matched text 6.453% (95% CI 6.172%, 6.745%), and visual prefill 38.511% (95% CI 33.331%, 47.091%). Visual prefill also raises measured shared-collective communication by 749.340% at the actual mixed iteration, versus 613.210% for token-matched text.

The token-matched control determines whether interference is visual-specific or generic to a same-sized prefill burst. This report does not infer communication causality merely from a slow visual prefill. Instrumentation bookkeeping overhead: 0.017% (PASS; same-process paired wrapper bypass).

# MLLM Matched Analysis

The replay used a real GQA image request capture containing 287 actual model-input tokens: 260 Vision and 27 Text, with exact hidden tensors sampled from the Qwen3-VL model. To isolate modality composition at equal execution volume, each modality-specific base distribution was repeated to the same target token count; this is a controlled activation-distribution replay, not a new natural request.

Layer 24, CP4+EP4, 30 repetitions:

| Length | Base distribution | Layer wall (ms) | Attention (ms) | MoE (ms) | CP boundary (ms) | Mean fanout |
|---:|---|---:|---:|---:|---:|---:|
| 32K | Mixed/all | 10.583 | 5.700 | 4.891 | 1.923 | 3.617 |
| 32K | Vision | 10.626 | 5.709 | 4.924 | 1.949 | 3.631 |
| 32K | Text | 10.568 | 5.771 | 4.774 | 1.975 | 3.740 |

At 32K, the layer wall varies by at most approximately 0.55%; there is no material modality-dependent CP degree, ready spread, or handoff mass signal. An 8K text-base run was slower, but it repeats only 27 text vectors rather than 260 vision vectors and was collected in a separate restart, so it is treated as an artificial periodicity/common-state confound rather than evidence.

Decision: no MLLM-specific claim. The negative CP-to-EP handoff result is generic to the tested Qwen MoE execution geometry. Kimi validation was not run because Qwen failed the performance/correctness gates.

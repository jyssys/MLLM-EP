# RQ4 — Visual versus token-matched text root cause

The comparison uses 10 paired clean repeats aligned to the trace-confirmed caller-visible mixed iteration, plus separate physical DeepEP traces.

| Metric | Visual 2,029 tokens (1,980 visual) | token-matched text 2,033 tokens |
|---|---:|---:|
| logical dispatch bytes | 1445330944 | 1450332160 |
| routing HHI | 0.008374 | 0.008626 |
| MNCD dispatch | 34.472 ms | 6.322 ms |
| CIDD | 6.323 ms/layer [6.176, 6.547] | 5.607 ms/layer [5.551, 5.700] |
| response peak inflation | 40.237% [28.871, 47.891] | 6.158% [-2.106, 10.627] |
| exact-hidden MoE-only replay dispatch/layer | 0.580 ms [0.533, 0.645] | 0.617 ms [0.514, 0.747] |
| exact-hidden MoE-only replay MoE/layer | 1.650 ms [1.501, 1.848] | 1.763 ms [1.439, 2.136] |

Exact-hidden replay status: `MEASURED_TRUE_EP4_EXACT_HIDDEN_ROUTE_REPLAY`. The paired visual-minus-text replay dispatch difference is -0.036 ms/layer [-0.214, 0.056]. Each replay clones the actual current layer hidden tensor, top-k IDs, and weights before the native call, executes one additional physical TRUE-EP4 DeepEP MoE call, discards its output, and leaves model semantics unchanged.

Root-cause classification: **MULTIMODAL_COMPUTE_OR_SCHEDULING_DOMINANT**. A V-CACHED end-to-end arm was not fabricated because the installed runtime exposes no validated external multimodal-embedding cache contract. Request preprocessing is timed before admission and is outside VSEL; vision-encoder work after admission remains part of the clean mixed iteration.

G4: **MLLM_STRESS_REGIME_ONLY**.

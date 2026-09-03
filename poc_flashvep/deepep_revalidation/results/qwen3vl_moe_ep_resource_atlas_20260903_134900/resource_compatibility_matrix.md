| pair | RESOURCE_COMPATIBILITY | DEPENDENCY | OVERLAP_CANDIDATE | reason | confidence | evidence |
|---|---|---|---|---|---|---|
| VISION_ENCODER+DEEPEP_DISPATCH | LOW | CROSS_REQUEST_INDEPENDENT | NO | Prior paired real run: wall slowdown 12.4%, communication slowdown 19.0%; not a positive candidate. | HIGH | MEASURED_NEGATIVE |
| VISION_ENCODER+DEEPEP_COMBINE | LOW | CROSS_REQUEST_INDEPENDENT | NO | Prior paired real run: wall slowdown 5.0%, communication slowdown 14.0%; not a positive candidate. | HIGH | MEASURED_NEGATIVE |
| VISION_ENCODER+EXPERT_GEMM | LOW | CROSS_REQUEST_INDEPENDENT | NO | Prior paired real run: wall slowdown 8.9%; compute contention. | HIGH | MEASURED_NEGATIVE |
| VISION_ATTN+DEEPEP_DISPATCH | MEDIUM | CROSS_REQUEST_INDEPENDENT | MAYBE | Different request and communication phase; resource complementarity is plausible but unvalidated. | LOW | INFERRED |
| VISION_MLP+DEEPEP_DISPATCH | MEDIUM | CROSS_REQUEST_INDEPENDENT | MAYBE | Compute plus communication may be complementary; HBM/SM contention is unknown. | LOW | INFERRED |
| LLM_ATTN+DEEPEP_DISPATCH | MEDIUM | HARD_DEPENDENCY | NO | Same request has ordering dependency; cross-request only is conditional. | LOW | INFERRED |
| LLM_ATTN+DEEPEP_COMBINE | MEDIUM | HARD_DEPENDENCY | NO | Same request combine follows expert/dispatch dependencies. | LOW | INFERRED |
| TP_COMM+DEEPEP_DISPATCH | LOW | HARD_DEPENDENCY | NO | Potentially shared communication fabric and ordering. | LOW | INFERRED |
| CPU_SCHEDULER+DEEPEP_DISPATCH | MEDIUM | CROSS_REQUEST_INDEPENDENT | MAYBE | CPU orchestration can overlap only if it does not introduce queueing jitter. | LOW | INFERRED |
| DECODE_ATTN+DEEPEP_DISPATCH | MEDIUM | CONDITIONAL | MAYBE | Cross-request independent, but both may compete for GPU memory/SM resources. | LOW | INFERRED |
| VISION_MERGER+DEEPEP_COMBINE | MEDIUM | CROSS_REQUEST_INDEPENDENT | MAYBE | Short projector/merger unit; requires bounded validation. | LOW | INFERRED |

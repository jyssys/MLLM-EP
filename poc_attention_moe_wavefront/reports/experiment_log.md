# Experiment log

| Run | Purpose | Result |
|---|---|---|
| baseline_r0 | 3 warmups; 10 clean + 10 instrumented repetitions/workload | Clean TTFT obtained; first flush path did not persist worker stage rows. |
| baseline_r1 | Stage persistence repair; 2 warmups; 6+6 repetitions | Rank 0/1 stage rows and clean request metrics valid. |
| baseline_r2 | Four-rank persistence validation; 1 warmup; 3+3 repetitions | All EP ranks persisted; runtime path reconfirmed. |
| N0_stock | Matched unsplit physical control | 2 clean CUDA repetitions/workload used after one timeline repetition. |
| N1_sequential | Exact 1/3 prefix/tail, one owner, no overlap | Median prefill tax +40.37%; greedy outputs agree. |
| N2_concurrent | Exact 1/3 prefix/tail, two streams, layer dependency | Median prefill change +479.76%; method logits drift; post-run auxiliary flush hung after primary data was saved and was interrupted. |
| split_scaling | 126 cut/workload cases, two reps each | Linear scaling rejected; Attention and MoE fragment ratios both about 2.41x. |

The N2 engine emitted every scheduled output and saved primary forward/layer
CUDA records and logits before the auxiliary shutdown flush hung. Only those
completed records are analyzed; the missing add-on stage trace is not imputed.


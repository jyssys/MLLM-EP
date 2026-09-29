# RQ1 — Direct communication-induced decode stall

Evidence is physical TRUE TP2-DP2-EP4 DeepEP on GPUs 2,3,6,7. Clean caller-wall timing and CUDA-event trace are separate. Iteration zero is the first trace-confirmed mixed prefill/decode iteration.

| Metric | Result |
|---|---:|
| Isolated decode dispatch / MoE layer | 0.397 ms |
| Visual mixed dispatch / MoE layer | 6.770 ms [6.649, 6.959] |
| Visual mixed MNCD dispatch | 34.520 ms |
| Visual CIDD / MoE layer | 6.323 ms [6.176, 6.547] |
| Visual CIDD / affected victim iteration | 303.492 ms [296.471, 314.272] |
| Peak VSEL | 68.167 ms [45.730, 85.810] |
| Peak VSEL percent | 46.123% [28.871, 61.608] |
| Decode-only response peak | 155.705 ms |
| Visual response peak | 218.176 ms |
| Response-window peak inflation | 40.237% [28.871, 47.891] |

`CIDD` is the measured critical-rank completion excess of the resident decode's physical shared DeepEP dispatch relative to its aligned isolated dispatch. No host timestamp is relabeled as collective wait time; CWT is `UNAVAILABLE_NO_RELIABLE_GPU_READY_TIMESTAMP`. MNCD is a non-preemptible communication quantum, not an assertion that a victim waited exactly that duration.

RQ1 verdict: **DIRECT_COMM_STALL_CONFIRMED**.

# Limitations

1. The tested model is Base Ouro-1.4B, BF16, Static R4. This is not a Thinking-model or dynamic-exit study.
2. Eight devices run independent single-GPU experiments. They do not form one TP, PP, or distributed-serving engine.
3. The pinned RLT path is the baseline. No production native-vLLM performance equivalence is claimed.
4. Diagnostic eight-worker throughput was not neutral on every GPU. Quiet confirmation is a separate lane.
5. Service events exclude earlier request-dependency waits. They can still contain CPU launch gaps and concurrent-stream contention.
6. The cost model does not establish a hard occupancy or deadline bound. Its Atomic blind exceedance was above the 5% screening target.
7. Only three source families contribute the single-prompt blind prediction targets. Thousands of quanta are not independent examples.
8. The optimizer utility pays remaining R4 stream service, but not an explicit model of all CPU work or future decode interruptions.
9. The first TRAIN processes used the documented planned-completion diagnostic label. Quiet confirmation uses the corrected frozen version.
10. Adaptive policies use one prefill span per batch. Native Mixed retains stronger co-batching capabilities.
11. Tight-budget C-only infeasibility and its smallest-progress fallback can create large relative gains against a weak operating point.
12. A gain over constrained C-only is insufficient if tuned G-only, fixed controls, or Mixed remain better.
13. The primary serving shapes are P2033/N4 and P8177/N8. P16K serving and broad queue/concurrency scaling are NOT_RUN.
14. Finite cohorts do not establish sustained open-loop capacity. No maximum sustainable request rate is measured.
15. Matched natural-output differences require accuracy evaluation. Isolated numerical parity does not prove universal batched parity.
16. The measured public quality pilot has 64 GSM8K and 50 MATH questions per arm. It cannot prove small population-level accuracy equivalence.
17. No hardware-counter attribution isolates HBM traffic, matrix utilization, or kernel-only busy time.
18. Binary variables, budget adaptation, loop yielding, and generic operator preemption are not automatically new contributions.

The paused native-vLLM Loop PP project remains paused.
No RL policy, learned depth gate, weight update, quantization, or new CUDA kernel is part of this study.

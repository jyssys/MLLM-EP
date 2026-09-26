# Clean actual EP4 baseline breakdown

The same actual TP2/DP2/EP4 full-model path produced the clean and diagnostic
runs. Clean wall time is authoritative; nested CUDA spans are not added as if
they were sequential barriers.

| Quantity | Value |
|---|---:|
| Input-ready first token | 2973.293 ms |
| Fixed decode 32 | 7150.996 ms |
| Fixed decode 128 | 16585.785 ms |
| Natural generation | 3220.822 ms median (112.301 output tokens mean) |
| Vision encoder diagnostic | 19.988 ms |
| Decoder diagnostic | 120.446 ms |
| Routed MoE outer span | 82.003 ms |
| Routed fraction of clean first-token wall | 2.745% |
| Profiler median/P90 overhead | 5.616% / 32.296% |

Shared expert is absent in this checkpoint. The large unattributed wall region
contains processor/H2D, scheduler/host gaps, LM head and other work not safely
separable with the bounded low-synchronization hook.

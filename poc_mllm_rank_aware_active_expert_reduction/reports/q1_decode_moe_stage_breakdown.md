
# Q1 decode MoE stage breakdown

Instrumented TRUE DeepEP runs were separated from clean timing. CUDA events bracket installed DeepEP prepare, fused Triton experts, and DeepEP finalize. Values are critical-rank means; diagnostic synchronization makes them unsuitable as TPOT.

| Topology | Batch | Dispatch ms | Expert ms | Combine ms | Total MoE ms | Dispatch share | Expert share | Combine share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EP2 | B1_trace | 0.2806 | 0.4221 | 0.1044 | 0.8070 | 34.77% | 52.30% | 12.93% |
| EP2 | B4_trace | 0.3477 | 0.4269 | 0.1113 | 0.8860 | 39.25% | 48.18% | 12.57% |
| EP2 | B8_trace | 0.3114 | 0.4695 | 0.1068 | 0.8878 | 35.08% | 52.89% | 12.03% |
| EP4 | B1_trace | 0.3686 | 0.4135 | 0.1279 | 0.9100 | 40.51% | 45.44% | 14.05% |
| EP4 | B4_trace | 0.4880 | 0.4291 | 0.1466 | 1.0637 | 45.88% | 40.34% | 13.78% |
| EP4 | B8_trace | 0.4361 | 0.4558 | 0.1463 | 1.0382 | 42.01% | 43.90% | 14.09% |

Mean shares: EP2 dispatch 36.36%, expert 51.12%, combine 12.51%; EP4 dispatch 42.80%, expert 43.23%, combine 13.97%.

Expert weights/HBM reads are charged to expert execution, never dispatch. Stage traces contain 1,392 EP2 and 2,784 EP4 rank-local invocations.

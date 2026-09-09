# FastPP materiality screen

The original dense method has arrival-regime dependence: ALP improves steady
requests but can lose on bursty inputs. P2P-enabled controls preserve the
direction. Existing greedy is a strong control and often recovers most of the
loss; the dense four-policy additional E2E envelope is only 0.25–0.64%.

Native Qwen3 MoE PP4 broad-screen additional existing-policy envelope is
0.59% request-mix / 2.35% equal-workload. Three held-out blocks do not show
stable material selection benefit. This is not proof against arbitrary chunks
or partitions. The completed full-workload-warmed five-knob control has **0%**
additional finite-policy mean-request E2E envelope, with PP-only best for both
steady and bursty. ALP is worse in all three paired restarts (median -21.54% and
-19.24% reduction, respectively). A static existing option resolves this observed
loss; it does not establish a non-trivial successor problem.

The most decisive completed causal attack is the existing uneven partition:
an apparent 23–26% stage-makespan opportunity becomes **14.91% steady / 11.98%
bursty request E2E regression**, median paired over three restarts, all pairs
worse. Same arrivals, same warmup and common KV capacity were used. This kills
the naive cost-proxy-to-request-gain inference, not the entire joint problem.

No material MLLM-specific failure is established. Native VL/EP support and
long-generation quality certification remain distinct limitations. EOS/KV
compatibility repairs are trivial engineering, excluded from successor novelty.

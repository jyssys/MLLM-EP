# Prior boundary

Source inspection date: 2026-10-18.
The pinned local implementation is the authority for executed behavior.
Published performance results do not transfer to this server.

| Source | Boundary for this study |
|---|---|
| [Ouro project](https://ouro-llm.github.io/) and [paper](https://arxiv.org/abs/2510.25741) | Supplies pretrained recurrent weights and the original gate. Weight reuse alone does not establish DP scheduling headroom |
| [CDB paper](https://arxiv.org/abs/2608.09444) and [official code](https://github.com/kschwethelm/continuous-depth-batching) | Supplies local loop-level batching/refill. The study does not claim local CDB as a new method |
| [vllm-rlt support table](https://github.com/ThinkFlowLab/vllm-rlt/blob/main/README.md) | Distinguishes base adaptive support from bounded Thinking fixed-depth validation. The study audits this checkout but does not execute it for primary measurements |
| [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) and [authors' code](https://github.com/raymin0223/mixture_of_recursions) | Uses a separately trained routing architecture. It is not a drop-in gate or proof of safe Ouro prefill skipping |
| [Enabling Dynamic Computation in Looped LMs](https://arxiv.org/abs/2610.09013) | Identifies cache-policy and actual-execution issues. Gate labels alone do not prove skipped work |
| [Learning Unmasking Policies](https://proceedings.mlr.press/v306/jazbec26a.html) | Inspiration for a possible later policy study. Diffusion unmasking is not available future Ouro depth information |

The official CDB exit-recording utility can derive schedules from a full reference-output forward.
Its documentation states that those schedules do not exactly reproduce live early-exit decode.
This PoC does not use that utility as an online trace source.
It records the actual engine's `_record_exit` events during free-running generation.
No future output or replayed depth drives those gate decisions.

Cross-GPU initial placement would be distinct from local CDB only if an incremental physical benefit survives strong generic controls.
That benefit must exceed queue-, length-, and calibrated-cost-aware placement.
A perfect-depth oracle would remain a retrospective upper bound.
It would not establish a deployable policy or measured GPU speedup.
This PoC trains no RL policy.

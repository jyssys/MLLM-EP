# Root-cause matrix (live corrected measurements)

| Candidate | Evidence | Update | Rationale |
|---|---|---|---|
| Measurement artifact / cross-GPU subtraction | same-device events, Nsight kernels, replay | REJECTED as primary | Tail survives corrected event timing and appears as real `notify_dispatch`/DeepEP kernels. |
| Pre-MoE queue / previous-step backlog | baseline lagged-state medians; sync intervention | SUPPORT++ | Tail rows have higher previous MoE/dispatch duration; draining outstanding work removes extreme decode tails. |
| DeepEP dispatch event/stream dependency | stage localization + Nsight | SUPPORT++ | First divergence is dispatch; `previous_event` and comm-stream dependency are in the stock HT path. |
| Expert kernel / workspace | fixed-route isolated and controlled replay | WEAKENED | 100 exact-input isolated samples have median ~0.235 ms and no giant tail; controlled route expert CV 5.5%. |
| DeepEP combine | stage localization | WEAKENED | Decode giant tails have normal combine (~0.04--0.08 ms); combine spikes occur only in a few mixed large-prefill events. |
| DP/EP synchronization | cross-rank grouping + source semantics | SUPPORT | Both TP ranks in the affected DP group often diverge together; DeepEP peer notification/barrier is collective. |
| Global GPU interference | nvidia-smi + local trace | WEAKENED, not excluded | Physical GPUs 1--4 were isolated during runs; other physical GPUs had unrelated users, but no causal external signal was measured. |
| Intrinsic route shape | fixed route replay | REJECTED for giant tail | Exact route/input replay is stable while online fixed-shape requests have 10--1,944 ms dispatch tails. |
| Layer/M threshold | online traces across M=1,137,284,412,536,2048 | WEAKENED | Tails occur across multiple M and layers; M changes magnitude but does not explain the event-level wait. |

## Current causal chain

Asynchronous DeepEP work from a preceding serving step remains outstanding on a communication stream/peer-notification path → the next `dispatch(previous_event=...)` cannot make its output usable → same-device dispatch CUDA span inflates (often both TP ranks of one DP group) → expert and combine run at ordinary cost after release. A diagnostic pre-MoE device synchronization drains the outstanding dependency and cuts extreme tail frequency, providing the intervention link.

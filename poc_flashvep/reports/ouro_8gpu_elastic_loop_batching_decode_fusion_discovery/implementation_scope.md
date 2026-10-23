# Implementation scope

The new worktree starts from clean pinned RLT.
It does not reuse the old prefill G2/G3 runtime patch.
The old worktree and every previous report remain unchanged.

| Component | New change | Preserved component |
| --- | --- | --- |
| SchedulerConfig | A decode group length with default1 | Existing resource and policy settings |
| ScheduledRequest | Immutable selected group length | Request identity, phase, position, depth, and capacity snapshot |
| Scheduler.schedule | Pure-decode common-depth clipping | Native admission and phase selection |
| Sync/async progress | Advance by the submitted group length | CODA transitions, output placeholders, and native exit handling |
| Executor descriptor | Pass the actual group to ExecutionBatch | ModelRuntime, attention, KV metadata, and GPU kernels |
| Executor guard | Reject grouped adaptive or out-of-depth leases | Existing valid one-loop paths |
| Experiment dispatcher | Scoped workers and source snapshots | Audited queue locks, process ownership, and resume checks |
| Numerical harness | Request/token-aligned hidden, logit, and sampled-KV comparisons | BF16 model execution and native sampling |
| Mechanism harness | Ready-frontier barrier and legal paired interventions | Full recurrent work, producer events, and native CODA |

The runtime patch changes three upstream files.
`patches/grouped_decode.patch` contains the removable source diff.
There is no new attention kernel, four-loop graph compiler, serving engine, TP, or PP implementation.

The frontier barrier is a controlled decode experiment.
It is not an online ELB admission policy.
It delays already-prefilled rows until the whole cohort completes its first output readback and host commit.
Frontend delivery remains a separate timestamp outside the engine step.
This separates initial prefill-induced depth fragmentation from steady decode grouping.
It cannot support a complete-serving TTFT claim.

The paired intervention precomputes valid new-row hidden state and KV.
It releases recurrent eligibility at an observed old-row R1 device boundary.
It does not claim that a fresh API request finished its prefill at that boundary.
That distinction remains explicit in every raw intervention record.

All eight GPUs execute independent single-GPU processes.
The paused native-vLLM Ouro Loop PP project remains paused.

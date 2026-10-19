# Code scope and factorial switches

## Frozen source

The CDB source commit is `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
The experiment does not modify that checkout.
The model revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The model is Base `ByteDance/Ouro-1.4B`, not the Thinking checkpoint.

The [CDB paper](https://arxiv.org/abs/2608.09444) describes batching between recurrent steps.
The [authors' repository](https://github.com/kschwethelm/continuous-depth-batching) provides both local scheduling modes.
This experiment tests the additional effect of cross-GPU initial placement.
It does not propose a new CDB implementation.

The [paper's performance protocol](https://arxiv.org/html/2608.09444v2) replays recorded exits and predetermined output lengths.
Its shared-KV examples can mix different recurrent depths in one core batch.
This study instead uses live exits, natural outputs, and slot-homogeneous `last_exited` KV.
The selected synchronous path also omits the paper's asynchronous lookahead execution.
These differences prevent a claim that the experiment reproduces the paper's best performance configuration.

The [hybrid diffusion paper](https://arxiv.org/abs/2602.21760) supplies presentation context only.
No number or single-sample latency claim transfers from that paper into this DP cohort study.

## Native no-refill execution

`continuous_depth_batching/continuous_api.py::_run_no_refill_loop` drives A0, A1, and A2.
`_run_no_refill_decode_wave` constructs a bounded resident token cohort.
It executes recurrent steps only for survivors.
`_record_exit` removes a token after the original online gate decision.
The wave runs one coda batch after its surviving cohort drains.
It never replaces an exited token inside that recurrent wave.

The next outer iteration can admit more requests.
Thus no-refill does not mean that a finished request blocks admission forever.
It means that the current shrinking recurrent wave receives no replacement work.
The native `scheduler_refills` counter must remain zero.

## Native refill execution

`continuous_depth_batching/continuous_api.py::_run_scheduler_loop` drives A3 and A4.
`scheduler.py::schedule_next` selects local coda, prefill, or recurrent work.
`enqueue_depth` counts fresh tokens that join other queued recurrent work.
The same model runner executes the actual model operations.
The same `last_exited` cache propagates terminal depth KV to later depth slots.

The selected KV layout requires slot-homogeneous recurrent kernels.
Tokens at different recurrent depths cannot automatically share one core kernel batch.
Refill can therefore change batch fragmentation and coda launch frequency.
Its presence does not guarantee lower elapsed time.

## Global scope

The CDB scheduler holds one local cache and local request queues.
It does not read another GPU's queue.
It does not use a DP process group for cross-replica placement.
It does not migrate requests to another GPU.
Its classification in this experiment is `LOCAL_ONLY`.

The pinned source locations are:

- [Native mode selection](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/continuous_api.py#L432).
- [No-refill execution](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/continuous_api.py#L593).
- [Local recurrent batching](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/scheduler.py#L322).
- [Terminal KV propagation](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/continuous_api.py#L1210).

The outer controller performs DP placement.
The generic burst rule balances outstanding request counts with stable rank ties.
Because all requests arrive together, that rule reduces to round-robin assignment.
The rule is identical in A1 and A3.
The historical speed order cycles four domains.
Round-robin therefore assigns each recurring domain to the same rank.
This is a workload-order limitation, not a representative random-arrival guarantee.
The quality order also alternates domains before burst placement.
Quality timing is not the primary performance metric.
One fixed shuffled C32/C64 sensitivity tests the speed-order limitation.
Its seed is fixed before that sensitivity runs.

Hindsight uses recorded future decode recurrent work.
Total future work includes both future output length and exit depth.
This factorial does not identify a pure depth-only advantage over perfect length knowledge.
Deterministic LPT assigns the largest request to the currently least-loaded rank.
The incoming request index and rank ID resolve ties.
The controller still submits requests in their original order.
The LPT sort changes placement planning, not the destination queue's FIFO order.
A2 and A4 use exactly the same frozen assignment.
The controller never uses correctness labels for placement.
The reference run and offline planning time are outside the hindsight serving makespan.
This is a clairvoyant diagnostic, not a costed online prediction method.
It knows reference future traces, not counterfactual traces after changed batching.

## Orthogonality

Placement and local refill have different state and action spaces.
They are independent configuration switches.
Their performance effects need not add or multiply independently.
Changing placement changes local batch composition.
Changing refill changes the cost of the resulting composition.
The paired factorial interaction measures this performance dependence.

Let `p` select each request's owner rank.
Let `m` select the local refill mode.
Let `F_r(p, m)` denote rank r's observed completion time.
Then cohort makespan is `T(p, m) = max_r F_r(p, m)`.
Separate switches do not make this function additive.
The generated outputs and exit traces can also depend on `p` and `m` through changed batching.
The observed interaction therefore need not describe fixed-work scheduling cost alone.

## Historical comparison

The earlier C64 value of 4.41% described a reduction in mean P95 request latency.
It was not a cohort-makespan reduction.
The approximate 1.046× conversion concerns that same P95 statistic.
The earlier 12.77% estimate used a selected trace-derived placement and modeled P95 latency.
The new primary rule uses depth-sum LPT and measured cohort makespan.
These differences prevent a direct equality claim.

The campaign therefore preserves the old selected C64 placement as a named sensitivity.
It applies that placement to both no-refill and refill.
The sensitivity does not replace the frozen primary method after results become available.

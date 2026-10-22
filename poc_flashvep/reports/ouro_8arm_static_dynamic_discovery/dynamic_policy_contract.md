# Dynamic policy contract

The reference is the official CDB repository at commit `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
The model is base `ByteDance/Ouro-1.4B`. The checkpoint revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The candidate does not use Thinking weights or an external depth predictor.

## Gate

Each token starts with CDF zero and survival one.
Each completed loop evaluates the original trained gate.
The hazard is `sigmoid(gate_logit)` in the model's output dtype, then converted to FP32.
The policy accumulates `CDF += hazard * survival` and `survival *= 1 - hazard`.
The host stores CDF and survival as Python floating-point scalars, as the pinned reference does.
The FP32 hazard conversion does not imply an FP32 tensor accumulator for those host scalars.
The threshold is 0.5. The minimum depth is two. The maximum depth is four.
The first loop contributes to the CDF, although that loop cannot terminate the token.
Prefill always executes four loops. Only subsequent decode forwards can terminate early.

## Refill and packing

The selected CDB mode is synchronous no-refill.
The official `_run_no_refill_decode_wave` removes exited tokens without adding their successors to the current wave.
The custom Dynamic controller likewise waits for every active token in a cohort to exit before it injects any successor.
The same barrier applies to the primary controlled Static/Dynamic replay.
Requests can still occupy different PP stages within a wave. This is real request-level overlap.

The prototype does not reproduce CDB tensor packing.
It executes one request per stage call. The official production-oriented CDB control can batch four rows.
The prototype can calculate a local coda when a request exits, but it cannot inject that request's next token early.
The official no-refill engine batches coda after the wave drains. This physical scheduling difference remains explicit.
Static task PP retains its previous asynchronous request FIFO. Those timings are not a scheduler-matched No-refill comparison.
Use controlled replay for the common no-refill scheduling comparison.

## KV policy

The policy is `last_exited`. Each `(loop, physical layer)` has a separate KV slot.
When a token exits, its current K/V row fills that token's row in every deeper slot.
Historical prefix rows remain unchanged. The next token cannot read an unfilled row.
The policy does not mix CDB with another paper's exit or cache policy.

Layer PP retains each physical layer's four depth slots on its owner rank.
Each owner copies its local exit row to its deeper local slots.
Loop PP sends the current exit-loop rows to the deeper loop owners.
Those owners acknowledge publication before the next token starts.
TP4 retains four attention heads per rank. Each rank copies its local current rows.
All TP ranks receive the same exit decision before the next collective sequence.

## Evidence levels

CPU tests compare the policy against the pinned CDB implementation.
These tests do not validate GPU KV values or generated answers.
GPU probes must compare outputs, depths, and all 96 logical KV slots.
Forced-exit probes test transport and cache updates. They are not live dynamic accuracy results.
Only live gate runs can demonstrate actual depth distributions and compute skipping.

## Reference sources

- [Official CDB code](https://github.com/kschwethelm/continuous-depth-batching).
- [CDB paper](https://arxiv.org/abs/2608.09444).
- Local policy: `src/looped_cdb/continuous_depth_batching/exit_policy.py`.
- Local cache updates: `src/looped_cdb/continuous_depth_batching/continuous_api.py`.

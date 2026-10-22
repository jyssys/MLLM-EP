# Dynamic reference validation

## Scope and status

This file records the bounded GPU validation before the Dynamic quality campaign.
It does not certify long free-running output parity or benchmark accuracy.
The task campaign reports those results separately.

The reference is the official CDB code at `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
The model is base `ByteDance/Ouro-1.4B`, not the Thinking variant.
Its checkpoint revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
All comparable inference uses BF16.

## Original weights and gate

The reference pilot checked all 269 original checkpoint tensors against the pinned safetensors file.
Every original tensor matched bitwise.
Those tensors occupy 2,869,305,346 bytes.
See [the startup audit](raw/CDB_SD_pilot_v1/weights_validation.json).

The adapter also creates optional `lookahead_exit_gate` and `preloop_exit_gate` tensors.
These optional gates do not replace the trained original gate.
The loader asserts that the selected module is `model.model.early_exit_gate`.
It checks both original gate tensors against the checkpoint.
The selected configuration disables lookahead and preloop exit.
See [the loader](../../../poc_decode_first_depth_aware_dp/runtime.py).

The policy uses the original cumulative hazard calculation.
Each decode token starts with CDF zero and survival one.
The first loop changes the CDF but cannot terminate the token.
The threshold is 0.5, with minimum depth two and maximum depth four.
Prefill always executes R4.
No gate training or threshold search uses task answers.

## Actual GPU probes

All four-rank jobs used `CUDA_VISIBLE_DEVICES=4,5,6,7`.
The Single jobs used physical GPU4.
The bounded synthetic prompts contain 128 tokens each.
Each synthetic request generated eight tokens.
The first output token comes from full-depth prefill.

| Comparison | Probe | Output IDs | Exit depths | KV evidence |
|---|---|---|---|---|
| Official CDB vs custom SD | Four live-gate requests | 4/4 exact | 4/4 exact | Reference current-row copies checked bitwise. Cross-backend full-KV equality was not measured. |
| SD vs LD | Four forced mixed-exit requests | 4/4 exact | 4/4 exact | All 96 logical slots matched bitwise. |
| SD vs OD | Four forced mixed-exit requests | 4/4 exact | 4/4 exact | All 96 logical slots matched bitwise. |
| SD vs LD | Four live-gate requests | 4/4 exact | 4/4 exact | All 96 logical slots matched bitwise. |
| SD vs OD | Four live-gate requests | 4/4 exact | 4/4 exact | All 96 logical slots matched bitwise. |
| SD vs TD | Four forced mixed-exit requests | 4/4 exact | 4/4 exact | Numerically unresolved. All four sharded head sets were assembled for comparison. |
| SD vs TD | Four live-gate requests | 4/4 exact | 4/4 exact | Numerically unresolved. Every TP rank checked its own current-row copies bitwise. |

The forced schedule rotates R2, R3, and R4 by request and decode position.
It tests simultaneous mixed depths and deeper reads after an earlier exit.
It is a transport and cache diagnostic, not a task-accuracy experiment.
Live-gate probes use no forced exit schedule.

The four live SD requests contain 28 decode tokens after prefill.
Their exit histogram is R2=18, R3=10, and R4=0.
They executed 66 decode loop invocations instead of 112 Fixed-R4 invocations.
This is 41.0714% fewer decode loop invocations.
It is work accounting, not a measured latency reduction.
Independent stage-body counters exclude graph preparation.
The heavy profiling lane must independently check the GPU execution pattern.

## TP warning

The live TD probe matched the short output and exit traces.
Its assembled KV tensors did not match the Single tensors numerically.
The aggregate relative L2 was 0.0533463462 for probes 0 and 2.
It was 0.0761231817 for probes 1 and 3.
The worst single-slot relative L2 values were 0.2152278870 and 0.2906669676.

These values do not satisfy a 0.03 KV diagnostic threshold.
They use all four head shards, unlike the earlier rank-0-only owner-KV metric.
Those different measurement scopes must remain separate.
The historical strict TP gate remains failed.
TD quality and timing therefore remain `NUMERICALLY_UNQUALIFIED_EXPLORATORY`.
Short token agreement does not remove this warning.

## Cache publication and scheduling

Every logical `(loop, physical layer)` owns a separate KV slot.
There are 96 logical slots per request.
An early exit copies only the current position into deeper slots.
It does not replace historical prefix rows.

LD copies each owner's local layer rows.
OD transfers the exit-loop rows to the later loop owners.
Those owners acknowledge the rows before the successor token starts.
TD copies each rank's four local attention heads into its deeper slots.
All TP ranks receive the same exit decision before the next collective sequence.

The primary controlled scheduler uses synchronous no-refill waves.
No successor token enters until all tokens in that wave finish.
The custom stages execute one request at a time.
The official reference can pack up to four rows.
It also batches coda execution after the wave drains.
Those backend and packing differences prevent a fully engine-matched cross-backend timing claim.

## Fixed regression before the GPU probes

The CPU regression found a stale valid-length field on an OD cache owner.
The owner received the new row but retained the previous valid length.
The fix updates that field after row publication.
The test failed before the fix and passed after it.
Subsequent physical forced-exit and live-gate probes matched all 96 slots bitwise.
The failed test record remains in [the regression artifact](numerical/CPU_KV_valid_length_regression.json).

## Raw evidence

- [Reference output and depth comparison](numerical/dynamic_reference_comparison.json).
- [LD forced-exit comparison](numerical/probe_probe_LD_forced_v1.json).
- [OD forced-exit comparison](numerical/probe_probe_OD_forced_v1.json).
- [LD live-gate comparison](numerical/probe_probe_LD_live_v1.json).
- [OD live-gate comparison](numerical/probe_probe_OD_live_v1.json).
- [TD forced-exit comparison](numerical/probe_TD_forced_v1_vs_SD.json).
- [TD live-gate comparison](numerical/probe_TD_graph_v1_vs_SD.json).
- [Exact policy and scheduling contract](dynamic_policy_contract.md).

All listed probe results are `LIVE_MEASURED`.
The count reduction is executed-work accounting from those probes.
No offline oracle replaces these probes.
No benchmark-quality equivalence follows from this bounded validation.

## Follow-up natural pilot: cross-backend drift

Both Single-Dynamic engines then completed the same 114 canonical-input pilot questions.
The official CDB reference and the custom SD prototype do not have long-generation token or exit-trace parity.
The two kernels and cache layouts remain different.

| Dataset | Official CDB correct/N | Custom SD correct/N | CDB minus SD accuracy, pp [paired 95% CI] | Exact output requests | Exact depth-trace requests |
|---|---|---|---|---|---|
| GSM8K | 43/64 (67.1875%) | 44/64 (68.7500%) | -1.5625 [-9.3750, 4.6875] | 41/64 | 17/64 |
| MATH | 33/50 (66.0000%) | 31/50 (62.0000%) | 4.0000 [-6.0000, 16.0000] | 8/50 | 0/50 |

The intervals use question-paired bootstrap with 5,000 draws and seed 20261021.
They do not establish accuracy equivalence.
Some identical output sequences have different exit decisions.
Longer divergent token histories can also change later gates and KV states.
These effects prevent a claim of task-wide CDB behavioral reproduction by the custom backend.
The custom eight-arm comparison retains its own Single-Dynamic reference and explicit exploratory backend status.
The original CDB reference remains a separate control.
The bounded four-request validation above remains unchanged.

See `behavioral_pair_summary.csv`, `behavioral_pair_details.csv`, and `paired_accuracy_deltas.csv`.
These pilot values do not replace the user-approved 400-question target for the custom eight arms.

The recorded first-depth-drift analysis found 97 requests with different depth traces.
For 67 requests, that first depth difference preceded any different input-token history.
For 36 of those requests, both engines recorded the CDF at the same decision loop.
All 36 recorded pairs lie on opposite sides of q=0.5.
For example, GSM8K question 0 recorded 0.5016326904 versus 0.4988670349 at generated-token index 20.
That request still had identical generated token IDs.
This is direct evidence of threshold sensitivity, not proof of one faulty upstream kernel.
The reference saves only its final exit CDF.
Earlier missing CDF values remain unavailable.
See `gate_margin_first_drift.csv` and `state/gate_margin_analysis.json`.

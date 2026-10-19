# Quality and exactness checkpoint

This report is a baseline audit, not a candidate-quality result.
The untouched campaign held-out cohort remains unused.
No candidate receives an accuracy PASS from these probes.

## Frozen protocol

The protocol is `artifacts/probe_protocol.json`.
It fixes eight calibration requests per domain and 32 held-out requests per domain.
The domains are GSM8K and MATH-500.
The metadata-only selection excludes the earlier factorial cohort and quality pilot.
The prompt templates, EOS, stops, cap, seed, and checkpoint revisions remain fixed.

Primary generation is greedy and free-running.
The cap is 8192 new tokens. EOS and frozen stop sequences can end generation earlier.
Only a native length limit counts as truncation.
The parser source is the preserved `poc_async_recurrent_pp/scoring.py`.
The analysis stores its SHA256.

The discovery margins are 1 percentage point for GSM8K and 2 points for MATH-500.
Eight requests per domain cannot establish either margin.
The small calibration cohort can reveal collapse or implementation mistakes.
It cannot prove benchmark equivalence.

## Distinct baseline implementations

| Path | Fixed R4 | Adaptive | State policy | Current evidence |
|---|---|---|---|---|
| Pinned native vLLM | All four passes | Not implemented in this path | Full depth-indexed KV | New TP1/2/4 controlled work and small numerical captures |
| Pinned CDB original gate | q1, minimum R4 | q0.5, minimum R2, maximum R4 | Full prefill and `last_exited` decode | Both-size tiny checks and natural calibration pilots |
| Public HF implementation | Complete configured recurrence | Computes configured passes before output selection | Explicit depth-indexed cache | Source audit, not a new official-HF task benchmark |

The native path uses a fused residual representation and combined projection kernels.
The staged reference uses its own operator order and kernels.
Both intend the same fixed-R4 mathematical function.
They do not produce bitwise-identical BF16 intermediates in the current evidence.
No report treats their cross-backend difference as a new method's gain or loss.

## Four-position numerical audit

The new capture uses identical input IDs and four controlled output IDs.
It saves full logits and selected hidden states.
It samples five cache positions, two global heads, and selected head features.
Every cache sample retains its logical layer/depth identity.
This is not a complete KV dump or a natural continuation test.

| Model | Native TP | Logit relative L2 vs staged | Sampled KV relative L2 | Top-1 matches |
|---|---:|---:|---:|---:|
| Base1.4B | 1 | 0.013239 | 0.016880 | 4/4 |
| Base1.4B | 2 | 0.013157 | 0.014638 | 4/4 |
| Base1.4B | 4 | 0.011161 | 0.014809 | 4/4 |
| Base2.6B | 1 | 0.018869 | 0.011976 | 4/4 |
| Base2.6B | 2 | 0.020804 | 0.012285 | 4/4 |
| Base2.6B | 4 | 0.017412 | 0.011558 | 4/4 |

Source: `artifacts/native_capture_checkpoint3.json`.
Relative L2 uses the staged-reference norm as its denominator.
These quantities are `QUALITY_ONLY`.
No post-hoc tolerance turns them into bitwise or benchmark equivalence.

## Natural baseline divergence

Native Base1.4B TP1 and staged fixed-R4 use the same 16 calibration requests.
Their raw token sequences match for five requests.
The other eleven requests have at least one raw token or length difference.
Some raw differences can concern terminal EOS or stop handling.
The analysis also records equality after removal of a terminal EOS token.
The preserved raw outputs determine which differences concern substantive continuation.

Several natural MATH responses diverge well before termination.
Thus four matched top-1 positions do not imply full natural-output parity.
Accuracy and output parity remain separate metrics.
The sample is too small to attribute an accuracy difference to a specific kernel or runtime component.

The generated summary is `artifacts/quality_calibration_checkpoint3.json`.
The Base2.6B fixed and native pilots are still active at that summary's cutoff.
They are absent from completed-arm summaries until their status file records completion.
Partial outputs remain available for diagnosis, not as completed benchmark rows.

## EOS accounting correction

The first native writer classified an unset stop reason as a text stop.
The pinned vLLM scheduler leaves `stop_reason` unset for its EOS path.
Explicit text or token stops populate that field.
The analysis derives EOS from the original native fields and preserves the writer's original classification.
It does not rewrite raw generation files.
The updated writer changes only metadata classification, not generation or sampling.

## Gate arithmetic is part of correctness

The selected CDB gate computes sigmoid in the BF16 signal dtype before promotion and host CDF accumulation.
An existing asynchronous helper instead promotes the logit before host sigmoid.
The two paths need not choose the same exit near q0.5.
The CPU counterexample is synthetic, not an observed model failure.
The full source and event contract is `reports/original_gate_overlap_contract.md`.

An exact overlap must transport the baseline-rounded hazard or establish an arithmetic-equivalent implementation.
It must preserve hazard order, selected hidden state, original gate tensors, and terminal-KV publication.
The current campaign does not enable the differing asynchronous helper.

## Candidate evidence boundaries

The finite-correction probe modifies private current-token scratch only.
It does not generate a candidate sequence.
Its logit KL, token agreement, and KV errors are diagnostics, not task accuracy.

The lossless byte transform reconstructs the tested BF16 bits.
It does not check a complete distributed codec or its runtime state versions.

The paged-depth fixture checks separate address ownership and FA3 compatibility.
It does not establish a new model runtime's quality.

No approximate mechanism currently qualifies for held-out quality confirmation.
The campaign retains the earlier stale/delta quality failure.
It does not repeat that method under a new architecture name.

## Completed baseline pilot update

`artifacts/quality_calibration_checkpoint4.json` supersedes the earlier incomplete-arm summary.
All four staged-reference arms and both native TP1 arms completed their 16 calibration requests.
The table preserves correct/N and request-bootstrap 95% intervals.
No candidate mechanism is active in these rows.

| Model | Runtime and mode | GSM8K | GSM8K 95% CI, % | MATH-500 | MATH 95% CI, % | MATH length-capped/N |
|---|---|---:|---|---:|---|---:|
| Base1.4B | Staged fixed R4 | 4/8, 50.0% | [12.5, 87.5] | 3/8, 37.5% | [12.5, 75.0] | 1/8 |
| Base1.4B | Staged adaptive | 5/8, 62.5% | [25.0, 87.5] | 4/8, 50.0% | [12.5, 87.5] | 0/8 |
| Base1.4B | Native fixed R4, TP1 | 3/8, 37.5% | [12.5, 75.0] | 4/8, 50.0% | [12.5, 87.5] | 0/8 |
| Base2.6B | Staged fixed R4 | 7/8, 87.5% | [62.5, 100] | 4/8, 50.0% | [12.5, 87.5] | 4/8 |
| Base2.6B | Staged adaptive | 6/8, 75.0% | [50.0, 100] | 4/8, 50.0% | [12.5, 87.5] | 3/8 |
| Base2.6B | Native fixed R4, TP1 | 7/8, 87.5% | [62.5, 100] | 5/8, 62.5% | [25.0, 100] | 3/8 |

The paired adaptive-minus-staged-fixed differences are operating-point context, not method effects.
For Base1.4B, both domains show +12.5 pp, with CI [0, 37.5] pp.
For Base2.6B GSM8K, the difference is −12.5 pp, with CI [−37.5, 0] pp.
For Base2.6B MATH, the difference is 0 pp, with CI [−37.5, 37.5] pp.

The native-minus-staged fixed differences also remain uncertain.
Base1.4B GSM8K shows −12.5 pp, CI [−37.5, 0] pp.
Base1.4B MATH shows +12.5 pp, CI [−25.0, 50.0] pp.
Base2.6B GSM8K shows 0 pp, CI [0, 0] pp for these eight paired outcomes.
That degenerate small-sample bootstrap does not establish population equivalence.
Base2.6B MATH shows +12.5 pp, CI [0, 37.5] pp.

Length-capped answers count as incomplete under the frozen rule.
The campaign does not increase the cap after these outcomes to improve the reported scores.
The wide intervals and truncation remain explicit limitations.
The held-out cohort is still untouched.

## Generic native option update

Existing native async scheduling improves controlled-work timing in the first Base1.4B TP2/TP4 controls.
This is baseline engineering, not a new recurrent architecture.
The comparison artifact is `artifacts/native_controls_checkpoint2.json`.
The TP2 four-position capture is bitwise identical in logits, hidden states, and sampled KV.
The TP4 capture contains numerical differences, so no bitwise claim applies there.
Neither result establishes full natural option-on/off output or accuracy parity.
The stronger conventional reference must retain these controls and their limits.
## Same-device scheduler-option recheck

The Base 2.6B natural-output recheck uses physical GPU7 for both arms.
The arms run in separate fresh processes with the same frozen sixteen calibration requests.
Both use the common 8192-token cap and original prompt, checkpoint, and evaluator.

Async OFF and ON produce identical raw output token IDs for all sixteen requests.
Each arm scores 7/8 on GSM8K and 4/8 on MATH-500.
The paired accuracy delta is 0 percentage points in each domain.
The empirical paired bootstrap interval is [0, 0] percentage points for each eight-item domain.
This degenerate small-sample interval does not prove population equivalence or a noninferiority margin.

The earlier cross-device comparison had a one-item MATH difference.
This recheck does not reproduce that difference on the matched device.
It does not identify its cause.
It does not establish TP4 natural-output parity or candidate quality preservation.
Natural latency cannot become a scheduler speedup when generated work differs.

Evidence: `artifacts/quality_option_same_gpu_recheck_v1.json` and the corresponding raw recheck directory.
The untouched held-out cohort remains unused.

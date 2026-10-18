# Stage 0: model, online gate, and KV validation

## Model boundary

The primary adaptive candidate is `ByteDance/Ouro-1.4B`, not the Thinking checkpoint.
Its revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The previous Thinking fixed-R4 result remains a separate anchor.
The study does not transfer its accuracy or adaptive-support claims.

The official CDB engine uses the original trained early-exit head.
No lookahead checkpoint, preloop checkpoint, or learned predictor supplies decisions.
The loader checks the selected module identity and the checkpoint gate tensors.
The native maximum depth is four.

The registered adaptive operating point is q=0.5 with minimum depth two.
This is a preregistered experimental operating point, not a claim of previously qualified accuracy.
The gate uses same-step cumulative hazard, without delayed consumption.

The CDB implementation applies sigmoid to its BF16 gate scalar, then converts that result to a host scalar.
It updates the token's survival probability and cumulative exit probability online.
The threshold test uses only completed work for that token.
The study does not tune q after observing benchmark answers.

## Actual execution: PASS in the bounded probe

The probe uses one 13-token prompt and a 16-output-token limit.
The first output token comes from full-depth prefill.
The remaining 15 positions use the staged decode path.

| Policy | Prefill recurrent token-steps | Decode depth histogram | Executed decode token-steps | Observed decoder-layer calls |
|---|---:|---|---:|---:|
| Fixed R4 | 52 | R4: 15 | 60 | 1,536 |
| Online q=0.5, minimum R2 | 52 | R2: 1, R3: 14 | 44 | 1,152 |

The layer observer counts actual eager model calls.
Each staged core invocation executes all 24 physical layers.
The observed totals equal `24 * (4 prefill passes + actual decode core launches)`.
This evidence distinguishes physical skipped execution from an exit label alone.

The decode token-step reduction is 26.667% against four loops for those same 15 adaptive decode positions.
It is not a measured latency reduction.
It is an existing local CDB/gate effect, not a new cross-GPU scheduling benefit.
The probe's 16-token cap is a correctness/execution diagnostic, not a quality evaluation.

## KV policy

The selected policy is `last_exited`.
It retains 96 independent depth/layer key buffers and 96 independent value buffers.
An early-exiting token copies its exit-depth KV into the deeper slots.
Later tokens therefore read defined past state at every supported depth.
The study does not use one shared KV slot.

The physical probe checks 360 exit-copy calls.
Every checked destination matches its source K/V rows exactly after completion.
The diagnostic synchronizes before it reads those rows.
It does not inspect a tensor from an unfinished producer kernel.
These copy checks add observer cost and do not support headline serving latency.

## Fixed-R4 comparison with the official HF release

The comparison uses the same pretrained checkpoint and prompt IDs.
It compares full logits at 16 teacher-aligned positions and every element of the final 96-slot KV state.
The official HF source remains unmodified.

| Quantity | Observed | Registered limit | Result |
|---|---:|---:|---|
| Maximum logit absolute difference | 0.25 | 0.5 | PASS |
| Mean logit absolute difference | 0.0236194 | 0.05 | PASS |
| Full-KV relative L2 difference | 0.00834413 | 0.02 | PASS |
| Teacher-aligned top-1 predictions | 15/16 match | Exact comparison, reported separately | PARTIAL |
| Natural 16-token output prefix | Differs from token index 3 | Exact comparison, reported separately | PARTIAL |

The teacher comparison follows the native fixed-R4 prefix.
Its 15/16 result is not 15/16 free-running token agreement.
The separate free-running trajectories differ after their first mismatch.
The source HF path uses Torch 2.8/SDPA, while CDB uses Torch 2.11/FlashAttention3.
The study does not assign the difference exclusively to one kernel without further evidence.

The fixed-R4 reference status is therefore `REFERENCE_PARTIAL`, despite the numerical-limit pass.
The study does not relax a limit or relabel the token mismatch as exact parity.
Attempts to align the old release with the newer reference environment encounter documented compatibility failures.
The study preserves those logs and does not patch RoPE to force a match.

## Quality diagnostic protocol

The quality pilot remains within Stage 0.
It does not qualify DP scheduling or oracle claims while reference parity is partial.
It measures free-running greedy generation, not teacher-forced answers.

The initial registered pilot has 16 GSM8K questions and 16 MATH-500 questions.
Question order uses a metadata-only hash before inference.
GSM8K uses the first three canonical CDB CoT examples.
MATH-500 uses a zero-shot prompt requesting a boxed final answer.
The base checkpoint uses no Thinking chat template.

The output cap is 4,096 tokens, with EOS and registered natural stop strings enabled.
The protocol predefines an 8,192-token truncation follow-up if incomplete output prevents interpretation.
Inputs and output limits remain identical between the two policies.
Correctness uses one common numeric/MathVerify extraction path.
Parsing failures and truncations remain in the denominator.

The registered final quality-loss targets are 1.0 pp for GSM8K and 2.0 pp for MATH-500.
The severe pilot stop requires at least 15 pp loss in either domain and a positive lower paired-bootstrap loss bound.
The bootstrap resamples question IDs, not tokens, layers, or loop calls.
The pilot cannot establish full-benchmark non-inferiority.
The measured quality table and final gate appear in `final_report.md`.

## Bounded pilot continuation

The initial MATH-500 point loss is 18.75 pp, with a 95% paired-bootstrap interval of [0.0, 37.5] pp.
That result does not satisfy the registered positive-lower-bound severe-stop rule.
It also does not establish an acceptable operating point.
The study extends Stage 0 to 32 questions per domain before any scheduling experiment.

The continuation uses the next questions in the original metadata-only order.
`artifacts/quality_protocol_v3_pilot32.json` records their IDs before continuation inference.
All thresholds, prompts, answer extraction, precision, and generation settings remain unchanged.
The study retains the first 16-question results and combines both phases by question ID.
It reports nominal intervals and the sequential look explicitly.
The additional consumed questions are not an untouched final held-out cohort.

## Final primary pilot quality

| Dataset | Fixed R4 | Adaptive | Accuracy drop, pp | Paired-bootstrap 95% CI, pp |
|---|---:|---:|---:|---|
| GSM8K | 22/32, 68.750% | 21/32, 65.625% | 3.125 | [-6.250, 12.500] |
| MATH-500 | 24/32, 75.000% | 19/32, 59.375% | 15.625 | [3.125, 28.125] |

These are free-running greedy, parsed-answer accuracy results with the common 4,096-token cap.
One extracted fixed-R4 MATH answer matches the gold answer, but its output reaches the cap.
The final report also gives completion-aware and longer-cap sensitivity results.
No question disappears from the denominator because of parsing failure or truncation.

The MATH-500 result satisfies the registered severe pilot stop.
Its point loss exceeds 15 pp and its nominal lower loss bound exceeds zero.
The Stage-0 primary decision is `QUALITY_FAIL` for this q=0.5/minimum-R2/last-exited configuration.
The study does not proceed to live DP4 placement, oracle scheduling, or predictor training.
It does not treat this result as proof that every adaptive operating point fails.

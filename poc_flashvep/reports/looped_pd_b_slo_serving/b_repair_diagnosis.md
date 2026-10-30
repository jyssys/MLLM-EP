# B correctness diagnosis

This document records diagnosis evidence. Read `repair_epoch_report.md` for the current qualification status.
A repair requires the full qualification. Small passing tests are insufficient.
The original15-hour clock and frozen SLOs remain unchanged.
The original B performance results remain rejected by their output gate.

## Actual failure and control

All counts below compare actual greedy-generated token sequences.
They are not answer accuracy measurements.

| Model | Workload | PD_G1 vs B | PD_G1 vs identical-control repeat | Evidence |
|---|---|---:|---:|---|
| Ouro R4 | Original W1,200 requests |185/200| Not this original pair | MEASURED |
| Huginn R32 | Original W1,200 requests |193/200| Not this original pair | MEASURED |
| Ouro R4 | Same-seed causal prefix,8 requests,cap128 |7/8|7/8| MEASURED |
| Huginn R32 | Same-seed causal prefix,33 requests,cap128 |31/33|30/33| MEASURED |

The causal prefixes preserve the original arrival seed and all eight warmup prompts.
They include every arrival through the earliest observed divergence, plus one request.
Ouro request3 diverged at output index125 in the original and fresh prefix tests.
The identical PD_G1 control also diverged at that index.
Thus, B is not necessary for this observed non-repeatability.
This does not exclude a second, independent transfer bug.

## Smaller reproducer

The seven-request Ouro prefix with cap126 produced6/7 exact outputs.
The four-request prefix produced4/4.
Five and six requests are not tested yet.
One- and two-request prefixes remove the target request3. They are negative controls, not complete minimizations.
Changing cap128 to126 can change scheduling and the reproduction rate.

The regression checks the actual serving engine outputs:

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-pd-abc/bin/python \
  -m poc_flashvep.looped_pd_b_slo_serving.repair_feedback \
  --left repair_r0_ouro_prefix8_PD_G1_repeat0 \
  --right repair_r0_ouro_prefix8_B_repeat1 \
  --expected 8 --result-name b_repair_live_red_before_fix
```

Observed result:7/8 exact. Exit status1 reports `EXACT_GENERATION_REGRESSION_FAILED`.

## Falsifiable hypotheses

1. Batch geometry changes BF16 arithmetic. Fixed matrix shapes should reduce differences if this mechanism dominates.
2. CUDA Graph buffers or replay ownership cause incorrect state. Disabling graphs should remove the failure if this mechanism dominates.
3. Stream, event, or asynchronous state ownership is incorrect. Serial execution or fewer streams should remove that failure.

Each probe changes one execution variable where possible.
Combined variants are separate diagnostic ablations.
No probe changes checkpoint weights, tensor precision, fixed depth, or the exact-output acceptance rule.
No diagnostic timing is accepted as clean serving performance.

## Scope of current evidence

Head-only fixed geometry did not remove every B/control mismatch.
Fixed decode-core geometry also did not remove every mismatch.
The combined decode-core and head variant did not pass all seven requests.
Disabling BF16 reduced-precision GEMM reductions did not pass all seven requests.
These failures exclude those individual changes as sufficient repairs.
They do not identify the first incorrect operation.

The current tests add full linear-operation geometry control and actual operand capture.
The capture records request positions, batch membership, hidden states, and logits.
It changes diagnostic timing. It cannot establish a serving speedup.

The first diagnostic launchers failed before generation because of Python import-path problems.
Those failures remain in the raw records. They are not model-correctness results.
Later `repair_v2_*` jobs use the corrected launcher.

## Acceptance after a possible repair

Keep the original failure evidence unchanged.
Register a new qualification epoch and the exact patch hash.
Apply the same repair to B and its matched controls.
Require the small regression, original200-request W1 outputs, fixed-depth counters, and transferred-KV checks.
Only then resume B serving measurements in that epoch.
Do not combine pre-repair and post-repair performance samples.
The official-HF discrepancy remains a separate limitation.

## Actual first-divergence capture

A late-only capture reproduced the original Ouro mismatch at output index125.
Both B/control and the identical-control repeat matched7/8 sequences.
The capture starts at decode step125. Earlier operands are not recorded in this test.

At step125, the captured R1 and R2 outputs match exactly.
The first observed recurrent-output difference occurs at R3, before the LM head.
The R3 input matches exactly. Its output relative L2 difference is0.0108540487.
The final R4 hidden-state relative L2 difference is0.0128933070.
These values identify the first observed boundary, not the first incorrect operator.
Historical KV differences could also affect that boundary.

The final logits differ by at most0.125 at this token.
The control chooses token4533 with a top-two margin of0.125.
B has a zero top-two margin. Greedy `argmax` chooses token30.
The recorded generated tokens agree with these argmax decisions.
The next token then uses different history. Its much larger logit difference is a consequence, not the initial cause.

`topk(2)` does not define the greedy tie winner. The analyzer uses `argmax` for the actual top-one token.
The raw logits remain unchanged.
Evidence: `b_repair_late_divergent_operands.csv` and its configuration summary.

## Geometry ablations and remaining uncertainty

The full-linear geometry probe passes7/7 B/control and7/7 control-repeat outputs in the small test.
However, another instrumented run still shows different internal states with that probe.
At decode step99, R1–R3 outputs match. R4 differs across batch sizes4 and3.
The final hidden relative L2 difference is0.0089044431. No greedy token differs in that captured pair.
Thus, full-linear padding alone does not establish batch-invariant internal execution.

The next candidate combines canonical linear geometry with the existing masked, fixed-capacity decode graph path.
Its first three seven-request runs also pass7/7 for both comparisons.
This is a candidate repair, not a full acceptance.
The qualification controller expands to8,32, and200 Ouro requests, then33 and200 Huginn requests.
It tests each model independently. A failed model does not block a qualified model.

Full operand capture can change scheduling enough to remove the original failure.
One such original-runtime capture matches all1512 paired operands.
That negative observation does not invalidate the separate late-only red reproduction.
The reports retain both results and their timing limitations.

The next KV capture compares the actual causal R3 prefix before the target token writes its current KV.
This distinguishes a changed historical KV prefix from a changed current hidden input.
It does not modify the frozen serving outputs.

### Causal-prefix KV capture result

The original-runtime capture again matches7/8 sequences and diverges at output index125.
Both runs have identical input tokens and identical generated history through index124.
The target request has59 prompt tokens. The captured R3 prefix excludes current position183.

| Runtime | Prompt KV exact | Prior decode KV exact | Different BF16 KV elements | Output sequence parity | Evidence |
|---|---|---|---:|---:|---|
| Original | Yes | No |551,940|7/8| MEASURED diagnostic |
| Canonical linear + fixed decode core candidate | Yes | Yes |0|8/8| MEASURED diagnostic |

The comparison covers17,989,632 KV elements at this target depth.
It does not cover every request, depth, or model.
The same prompt KV reaches decode in this reproduced case.
Differences appear in KV created during decode, despite identical earlier output tokens.
Thus, corrupted prompt-KV delivery is not necessary for this case.
The paired control repeat also fails without B.
Together, these observations support batch-dependent decode arithmetic as the working cause.
They do not prove that every observed mismatch has this cause.

The canonical candidate removes the captured KV difference in this small test.
Larger output qualification and reversed-fragment byte checks remain mandatory.
Padding can add real GPU work and memory traffic. Serving timings must include that cost.
The semantic full-depth row counter does not count dummy padded FLOPs.
The same canonical changes apply to every comparator in the new epoch.

Huginn also has attention-bearing coda layers.
The candidate uses the existing active-row KV guards for their fixed-capacity graph path.
Huginn's expanded output, KV, and performance results remain separate qualifications.
No Ouro diagnosis is automatically generalized to Huginn.

### Pre-repair baseline shift

The first canonical Ouro200-request PD_G1 run matches142/200 original PD_G1 sequences.
Its input IDs match200/200. Its output lengths match196/200.
This comparison uses the original arrival seed and prompt set.
Thus, canonicalization is not an exact restoration of the original run's outputs.
The original runtime also shows self-repeat differences. This comparison alone cannot assign all58 differences to the patch.
It does not measure an accuracy change.

The repair can qualify only matched-runtime B/control execution under these tests.
Official-model output fidelity remains a separate unresolved gate.
Fresh small Native/HF triage is scheduled after full qualification.
All post-repair serving results remain a separate, explicitly conditional execution epoch.

## Expanded output regression

The canonical Ouro candidate passes all three completed stages below.
Each stage uses a fresh B run and two fresh PD_G1 runs.

| Requests per run | PD_G1/B exact | PD_G1 repeat exact | Evidence |
|---:|---:|---:|---|
|8|8/8|8/8| MEASURED |
|32|32/32|32/32| MEASURED |
|200|200/200|200/200| MEASURED |

At200 requests, input IDs, output lengths, stop reasons, and recorded depth traces also match200/200 in both comparisons.
The explicit `repair_feedback` commands return exit status0 for both full200-request comparisons.
Their saved results are `configs/b_repair_ouro_full200_green.json` and `configs/b_repair_ouro_full200_repeat_green.json`.

Huginn also passes33/33 B/control and33/33 control-repeat sequences.
At2026-10-30 03:14 KST, its200-request regression is running. Do not substitute the pilot for that full gate.
The current controller states and `repair_epoch_report.md` record subsequent progress.

PyTorch documents that mathematically equivalent batched and sliced operations need not produce identical floating-point results.
This general fact is not proof of this bug's cause. [PyTorch2.11 numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)
vLLM treats batch invariance as a separate feature and notes its performance trade-off.
That page does not certify the private RLT engine or either model. [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)

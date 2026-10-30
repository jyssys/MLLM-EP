# Numerical correctness diagnosis for Ouro and Huginn PD

## Decision and scope

Ouro passes the tested internal parity qualification after a selective numerical correction.
Huginn's Down-plus-norm candidate fails the independent-seed and closed-loop qualification.
An additional adapter-specific correction is under independent qualification.
Neither result establishes lossless equivalence to the original HF model.
No large serving-capacity campaign was started.

The baseline commit is `fea20edef840ed357ee0493c2f0dd69668c66c1f`.
The branch is `flashvep/looped-pd-b-numerical-diagnosis`.
The installed RLT and imported LoopServing sources remain read-only.
All new corrections are opt-in, process-local patches in `looped_pd_b_numerical_diagnosis/`.
All model parameters retain BF16. Ouro uses fixed R4. Huginn uses fixed R32.

Evidence labels have these meanings:

- `MEASURED`: a real GPU execution or operator replay.
- `TRACE_ANALYSIS`: an analysis of saved, real request records.
- `REFERENCE_REPORTED`: a result stated in the imported reference, without its original raw files.
- `NOT_MEASURED`: no result is available.

## Frozen failures and fresh reproduction

The frozen failures remain unchanged.
`configs/failure_manifest.json` contains complete input IDs, output histories, sample IDs, seeds, and source hashes.
`audit.py --assert-parity` is the red-capable regression command. Its expected exit code is 1.
That command tests frozen evidence. It does not launch a model.

| Evidence | Model and runtime | PD_G1 repeat agreement | PD_G1 versus B | Failure details |
|---|---|---:|---:|---|
| TRACE_ANALYSIS | Frozen Ouro original prefix8 | 7/8 | 7/8 | Sample3, output125, prompt59 |
| TRACE_ANALYSIS | Frozen Ouro legacy canonical400 | NOT_MEASURED in that pair | 399/400 | Sample233, output4, prompt770 |
| TRACE_ANALYSIS | Frozen Huginn legacy canonical200 | 198/200 | 199/200 | Repeat:42/output31 and92/output71. B:92/output71 |
| MEASURED | Fresh Ouro original200, seed2026103001 | 181/200 | 179/200 | The original runtime is not repeatable |
| MEASURED | Fresh Ouro legacy canonical400, seed2026107001 | 399/400 | 398/400 | Repeat:33/output4. B:33 and233/output4 |
| MEASURED | Fresh Huginn original prefix93, cap72 | 92/93 | 91/93 | Not the same runtime as the frozen canonical failure |

Samples33 and233 repeat the same source prompt in the 400-request cycling workload.
The fresh legacy canonical run reproduces sample233/output4 in a real PD execution.
The teacher-forced probes below separately test the exact histories of frozen Huginn samples42 and92.
They do not claim to reproduce the complete old Huginn serving schedule.

## First observed internal differences

Teacher forcing uses the saved common history before the first divergent output.
It prevents a different earlier generated token from explaining a later tensor difference.
These runs use instrumented eager native operators.
They do not reproduce the instruction schedule of a CUDA-graph PD run.

Loop and layer indices below are zero-based.
The reports identify the first observed difference within the inspected operations and rows.
They do not claim to inspect every internal instruction.

| Model, actual failure history | Probe | First observed operation | Identical input | Different output elements | Maximum absolute error | Relative L2 |
|---|---|---|---|---:|---:|---:|
| Ouro sample3 | Original, selected prompt row | Loop0, layer0, `mlp.down_proj` | Yes | 9/2048 | 0.0001220703125 | 3.881517841364257e-06 |
| Ouro sample3 | Down-only, all decode steps | Input step51, loop3, layer20, `post_attention_layernorm_2` | Yes | 8/2048 | 0.001953125 | 0.0006544701172970235 |
| Huginn sample92 | Original, all decode steps | Input step22, loop14, `core_block.2.norm_1` | Yes | 28/5280 | 0.015625 | 0.000576566148083657 |
| Ouro sample233 | Legacy canonical, chunk256, tail2 | Prompt offset768, loop2, layer18, `input_layernorm_2` | Yes | 12/2048 | 0.000244140625 | 0.0003536526346579194 |

The later full-prompt trace for Ouro sample3 records267 differing Down elements across the inspected59-token sequence.
Its maximum error is0.001953125 and relative L2 is8.062752385740168e-06.
That population differs from the single selected row in the first table row.
The two denominators must not be combined.

The late instrumented real-PD snapshot compares30 matched stage-boundary records through output4 for sample233.
Those captured PD_G1/B hidden states and logits match exactly.
The actual captured output4 is354 in both arms.
The coda micro-probe is invariant across M1–8 on that captured operand.
Thus this late instrumented run does not reproduce the clean400-request failure.
It cannot identify coda or LM-head arithmetic as the cause of that clean failure.
The chunk-tail teacher replay supplies a separate same-input numerical counterexample.
See `configs/actual_PD_boundary_comparison.json`.

Raw float values and BF16 bit patterns are in `first_divergent_tensor_values.csv`.
The corresponding `.pt` files preserve common operands and both operator outputs.

### Why Down-only is insufficient in this RLT runtime

LoopServing's historical Down finding applies to its specific operator stack.
RLT uses a different PyTorch RMSNorm implementation.
Its FP32 `square().mean()` reduction can change with row geometry.
The resulting small FP32 difference can cross a BF16 rounding boundary.
Repeated recurrent operations can amplify that difference.

For the captured Huginn norm operand, the variance changes by1.1920928955078125e-07.
The normalized FP32 result changes by at most4.76837158203125e-07.
BF16 conversion produces a maximum difference of0.0078125.
The learned norm scale raises the maximum output difference to0.015625.

The Ouro operand has a variance difference of2.3283064365386963e-10.
Its inverse changes by at most1.9073486328125e-06.
These values are measured operator differences. They are not estimates of answer accuracy.

`norm_probe.py` compares24 row counts from1 through256, including every count1–16.
The fixed-reduction candidate matches its M16 reference on every row of those two captured operands.
That finite operator test is not a proof for all model inputs.

### Why the previous all-Linear patch was not invariant

The legacy correction uses8 rows for small calls and256 rows for larger calls.
Ouro sample233 ends its770-token prompt with a2-token chunk after three256-token chunks.
Changing the number of requests changes the tail's total GEMM row count.
The old correction can cross its8-to256 geometry boundary.

The chunked legacy replay predicts output4 as253,253,354,253 for batch1,2,4,8.
The unchunked replay does not reproduce this difference.
That negative control shows why the actual chunk tail matters.
Padding every Linear without one consistent geometry did not solve the problem.

## Selective correction and independent ablations

The main candidate is `down_norm` in `patches.py`.
It applies two changes:

1. Execute each actual MLP Down projection in uniform256-row tiles.
2. Use a16-row FP32 reduction for RMSNorm calls with fewer than16 real rows.

The correction does not pad KV entries or create model tokens.
It slices real outputs in their original order.
It retains Q/K/V/O, Gate/Up, adapter and LM-head implementations.
It retains the model weights, fixed recurrence, FA4 backend and existing graph mechanism.
The extra padded work is real GPU work. It is not free.

| Ablation | Measured result | Limit |
|---|---|---|
| Original | Reproduces numerical and free-running failures | Rejected |
| Down fixed only | Ouro norm counterexample remains. Huginn sample92 later differs in `prelude.0.norm_3` | Insufficient |
| Norm fixed only | Huginn sample92 batch1/2 inspected tensors match | Does not establish general Down invariance |
| Norm fixed only | Ouro sample3 still has a same-input Down difference | Insufficient |
| Down plus norm | Ouro samples3 and233 inspected tensors match across tested sizes | Finite coverage only |
| Down plus norm | Huginn historical samples42 and92 inspected tensors match across tested sizes | New qualification still fails |
| Down plus norm plus fixed-capacity core/coda guard | Four-request Huginn PD_G1/B and valid-prefix fingerprints pass | No200/400 qualification for this extra variant |

Ouro sample3 tests batch1/2/3/4/8/16 under the corrected replay.
Ouro sample233 tests batch1/2/4/8 with chunk256.
Huginn sample92 tests batch1/2/3/4 in the corrected all-step replay.
Huginn sample42 tests batch1/2.
The final greedy outputs match within each corrected probe.
Some final logit margins are exactly zero. The actual decision uses `torch.argmax`, not top-k tie order.

## Free-running qualification

Each arm starts a fresh process.
Every comparison uses the same prompts, seeds, model revision, fixed depth and output cap128.
Ouro uses rate1 request/s. Huginn uses rate0.2 request/s for the finite arrival tests.
These are finite correctness cohorts, not sustainable serving-capacity measurements.

| Model | Cohort | PD_G1 repeat | PD_G1 versus B | Decision |
|---|---|---:|---:|---|
| Ouro | N8, seed2026103001 | 8/8 | 8/8 | PASS within scope |
| Ouro | N200, seed2026103001 | 200/200 | 200/200 | PASS within scope |
| Ouro | N400, old failing seed2026107001 | 400/400 | 400/400 | PASS within scope |
| Ouro | N400, independent seed2026109001 | 400/400 | 400/400 | PASS within scope |
| Huginn | N8, seed2026103001 | 8/8 | 8/8 | PASS within scope |
| Huginn | N200, seed2026103001 | 200/200 | 200/200 | PASS within scope |
| Huginn | N200, independent seed2026109001 | 199/200 | 199/200 | NO_GO_CORRECTNESS |
| Huginn | N400, closed-loop, capacity8, seed2026109001 | 393/400 | NOT_RUN after repeat failure | NO_GO_CORRECTNESS |

Huginn's independent200-request repeat differs at sample85/output63.
Its B comparison differs at the same sample, output98.
The closed-loop repeat differs at samples15,56,85,175,256,285,381.
Two baseline starts use different physical GPU pairs in that400-request test.
Thus that test includes both scheduling and cross-pair variation.
It does not isolate one cause between them.

The400-request workload cycles200 source prompts twice. It does not contain400 unique prompts.
No new successful result replaces the frozen failures.
Huginn remains rejected by the requested full acceptance gate.

## Request order, KV and work accounting

Reversing the eight source requests and reducing capacity to3 preserves each source request's sampling seed.
Both models retain8/8 output agreement for PD_G1 and B against their original-order reference.
`order_seed_parity.csv` records source IDs, seed checks and comparison scope.
This small test does not override Huginn's larger independent-seed failure.

Prompt-prefix hashes compare valid depth/layer KV entries and final hidden states.
Ouro sample233 has exact KV and hidden hashes across batch1/2/4/8 after correction.
Huginn sample92 has exact hashes across batch1/2/4 after correction.
The original Ouro sample3 differs in both prompt KV and hidden hashes between batch1 and2.
This occurs without a PD transfer. It is a numerical counterexample, not evidence of NIXL corruption.

Actual B transfer fingerprint tests pass4/4 measured requests per model.
The queued-compute diagnostic passes8/8 measured fingerprints per model.
The guarded Huginn core/coda ablation passes4/4 for each PD_G1 and B arm.
Warmup fingerprints are excluded from those denominators.

Huginn coda prefix checks exclude the final prompt position before its first coda call.
Non-core KV layers use depth0 only.
The active-row guard preserves write masks, output slicing and real-row frontier updates.
These checks cover transferred valid prefixes. They do not exclude every possible later state or event defect.

Measured logical recurrent work equals:

`fixed_depth × sum(prompt_tokens + generated_tokens − 1)`.

All completed indexed PD qualification runs satisfy their own generated-work identity.
This counts real model rows. It does not count padded GEMM rows as additional tokens.
Huginn's failing runs have different generated lengths, so their total work can differ despite valid accounting.
The failures therefore cannot support a same-work performance claim.

## Original HF fidelity and answer accuracy

| Model | Fresh corrected native versus frozen official HF | Identical prompts | Interpretation |
|---|---:|---:|---|
| Ouro | 8/8 sequences | 8/8 | Small numerical triage only |
| Huginn | 5/8 sequences | 8/8 | Original-model fidelity remains unresolved |

GSM8K and MATH-500 answer accuracy are `NOT_MEASURED` in this debugging task.
Internal PD parity does not establish answer accuracy or lossless original-model equivalence.
The small HF comparison cannot establish an accuracy regression rate.

## Residual Huginn adapter counterexamples

The new failed200-request case is sample85, prompt59, seed2026109086.
The new failed400-request case is sample56, prompt62, seed2026109057.
Both use the same current router-free dense model and fixed R32.

| Actual failure history | First observed same-input difference after Down-plus-norm | Compared batch sizes | Different elements | Maximum absolute error | Relative L2 |
|---|---|---|---:|---:|---:|
| Sample85, history through output62 | Prompt, loop0, `transformer.adapter` | 1 versus4 or8 | 26/5280 | 0.001953125 | 0.000252425525104627 |
| Sample56, before first output | Prompt, loop0, `transformer.adapter` | 1 versus4 or8 | 42/5280 | 0.0009765625 | 0.00019381538731977344 |

Sample56 predicts3635 at batch1/2, but1401 at batch4/8 in the instrumented replay.
Its top-two margin changes from0.125 to0.
Thus the residual issue is not only activation similarity or an insignificant hidden-state difference.
It changes the greedy output on the same prompt and request seed.
No PD transfer is needed for this counterexample.

The earlier random-operand adapter microbenchmark passed its inspected geometries.
The real-operand counterexamples show why that result was not a proof of numerical invariance.
The adapter maps the concatenated state and injection to the recurrent state.
It is separate from the MLP Down projection.

The new `down_norm_adapter` candidate adds fixed256-row tiles only to `transformer.adapter`.
It retains the existing Down and norm corrections.
It does not pad Q/K/V/O, Gate/Up or the LM head.
It leaves the installed engine unchanged.

Both residual histories now match all inspected tensors across batch1/2/4/8.
Their valid prompt KV and hidden hashes are also recorded.
Sample85 predicts286 at every size. Sample56 predicts1401 at every size.
This is a passing minimal replay, not the final200/400 acceptance.

`adapter_qualification.py` runs the original and independent200-request seeds.
It also runs400-request closed-loop gates for seeds2026109001 and2026109002.
Four physical GPU pairs execute these numerical gates independently.
Their timing is not a clean performance measurement.
Each cohort compares an independent PD_G1 repeat before comparing B.
The controller stops that cohort if its baseline repeat fails.
Final acceptance is PENDING in `state/huginn_adapter_qualification.json`.

## Remaining work and stop condition

The residual Huginn cases received bounded teacher-forced probes in a new namespace.
The adapter-specific correction requires the expanded acceptance gates above.
No global Linear rewrite or large capacity sweep belongs to the current correction.
An unresolved operator or state difference remains a failure, not an accepted BF16 exception.
The final summary records the residual-probe outcome and the independent performance decision.

See [execution comparison](looped_pd_b_loopserving_comparison.md),
[raw qualification index](looped_pd_b_numerical_diagnosis/configs/results_index.json),
and [failure log](looped_pd_b_numerical_diagnosis/failures.md).

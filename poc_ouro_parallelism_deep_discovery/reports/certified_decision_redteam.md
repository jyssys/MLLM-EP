# C09: Certified partial projection and early decisions

Revision: 1.
Worker: `/root/independent_axes`.
Segment began: 2026-10-19 10:44:10 UTC.
The audit artifact records the end checkpoint.

## Outcome

A partial-dot norm bound can certify a greedy LM-head winner after the selected hidden state is ready.
The bound is generic exact maximum-inner-product search.
It does not certify an unfinished recurrent output from an earlier hidden state.
Such a claim needs a separate bound on the remaining core and its KV-dependent state.

Verdict: `GENERIC_ONLY` for the audited final-head mechanism.
Earlier-hidden continuation certification remains `CONDITIONAL_MECHANISM`, with no useful core bound established.
This result is not an impossibility theorem about every certified inference algorithm.
No model-quality or GPU test ran.

## 1. Actual decision dependencies

Both Base checkpoints use hidden width 2048 and vocabulary size 49,152.
Their LM head has no bias.
The 1.4B and 2.6B physical stacks contain 24 and 48 layers.
Both use fixed R4 in the fixed-depth comparison.

The official source executes the physical stack and loop RMSNorm before each gate.
The gate is an affine map from 2048 features to one scalar.
The selected hidden state enters the LM head.
Fixed R4 does not require intermediate LM-head projections.
Adding them for candidate ordering creates extra work.

The official HF forward computes all loops before selecting an output.
That behavior differs from physical skipping in the staged CDB runtime.
The CDB adapter executes one physical stack, applies loop RMSNorm, and then calls the original same-step gate.
Its coda is a pure LM-head projection.
The runner then applies `torch.argmax`.
The inspected cumulative-hazard policy consumes the sigmoid value and updates survival and the exit CDF.

The installed vLLM Ouro forward returns the normalized final hidden state after its recurrent loop.
Its `compute_logits` uses the logits processor.
That processor also defines `get_top_tokens` with local argmax and a small cross-rank winner exchange.
This generic method does not gather the full vocabulary.
The inspected Ouro method does not directly call it.
Its existence is a baseline requirement, not proof of reachability in this run.

| Decision | Required input | What a certificate can avoid | What it cannot avoid by itself |
|---|---|---|---|
| Fixed R4 greedy token | Final selected hidden state | Some vocabulary dot products | Earlier required recurrent passes |
| Adaptive greedy token | Selected hidden and completed exit routing | Some vocabulary dot products | Computing an unknown exit decision |
| Original current gate | Current loop-normalized hidden and policy state | A tail of one affine dot product | Producing the current hidden |
| Later recurrent greedy token | A valid enclosure of the later hidden | Work excluded by that enclosure | Arbitrary unfinished nonlinear/KV work |

The head and gate can use the same already-ready hidden.
Their arithmetic may overlap if resources permit.
A head computed before the gate selects that depth is speculative work.
It must be discarded if the gate continues.

## 2. Valid real-arithmetic head bound

Let `z_i = w_i^T h + b_i` for vocabulary row i.
Ouro uses `b_i=0`.
Partition the feature coordinates into S and its complement T.
All entries of the selected h must exist before their values enter this bound.

Define:

`s_i = w_i,S^T h_S + b_i`

`rho_i = ||w_i,T||_2 * ||h_T||_2`

`L_i = s_i - rho_i`, and `U_i = s_i + rho_i`.

Cauchy-Schwarz gives `L_i <= z_i <= U_i`.
The tail coordinates remain part of the current hidden, not an unknown later hidden.
Their norm must be known or conservatively bounded.

Compute one incumbent row c completely.
If `z_c > max_{i != c} U_i`, c is the unique greedy winner.
Alternatively, `L_c > max_{i != c} U_i` certifies c without a full incumbent dot.
Failure of this test means “unresolved,” not “wrong.”
The implementation can refine the bound or compute the remaining products.

The CPU harness avoids rounded square roots in its acceptance test.
For each competitor, it requires `z_c-s_i > 0`.
It then requires `(z_c-s_i)^2 > ||w_i,T||_2^2 * ||h_T||_2^2`.
Exact rational arithmetic evaluates these inequalities.
Strict inequalities avoid an unresolved tie policy.
A non-strict test must implement the baseline's exact tie rule.

The same incremental bound appears in FEXIPRO's section 2.2.1, equation 1, and Algorithm 2.
That source predates Ouro and this campaign.
[Li, Chan, Yiu, and Mamoulis, SIGMOD 2017](https://lihui.info/doc/SIGMOD17.pdf).

## 3. Costs that the bound adds

Let V=49,152, D=2048, m=|S|, and A count unresolved competitors.
An incumbent-first implementation performs `D + (V-1)*m + A*(D-m)` projection multiplications.
This assumes that fallback reuses every prefix product.
The dense reference performs `V*D = 100,663,296` multiplications per token.
Recomputing survivors with a standard dense kernel can repeat prefix work.
That repetition must enter the cost ledger.

Precomputing tail norms needs a complete weight read during setup.
The BF16 LM-head weights occupy 201,326,592 bytes, or 192 MiB, before TP sharding.
One FP32 norm value per row occupies 196,608 bytes.
Each additional split needs another table unless a different representation replaces it.
FP64 norm tables double this storage.
Safe finite-precision bounds also need outward rounding or certified metadata errors.

Online costs include:

- The tail hidden norm and its reduction.
- Prefix projection reads and arithmetic.
- Norm-table reads, interval arithmetic, comparisons, and incumbent selection.
- Survivor indices, compaction, irregular weight reads, and fallback kernels.
- Extra launch boundaries or device control.
- Cross-rank decisions when a global incumbent guides pruning.

The head weights divide into 96/48/24 MiB per rank at P2/P4/P8 under balanced vocabulary sharding.
The replicated hidden state occupies 4096 bytes per token in BF16.
Each rank can certify its local winner and exchange only a value/index pair.
A global-incumbent design can instead exchange an incumbent and unresolved upper bounds.
Both designs require synchronization and an explicit tie policy.
Repeated refinement rounds can add more synchronization than the one-round dense argmax control.

A generic vocab-parallel argmax already needs only small winner messages.
Thus, C09 cannot claim all full-vocabulary communication as a unique saving.
The installed source uses two FP32 values per token per rank for this exchange.
Actual collective wire bytes exceed the logical payload.

The critical-path test is:

`T_prefix + T_bounds + T_survivors + T_control + T_messages < T_matched_dense_head_and_argmax`.

The recurrent core remains unchanged.
If its exposed head fraction is f, complete head removal gives the conditional bound `S <= 1/(1-f)`.
This audit did not measure f.
Projection operation counts are not a substitute for that measurement.

For context only, fixed R4 dense core weights total 9408 MiB of pass-weight volume for 1.4B.
The corresponding 2.6B count is 18,816 MiB.
Adding one 192 MiB head gives head fractions of 2.0000% and about 1.0101%.
These are structural byte counts under an ideal single read per required dense operator.
They are not measured HBM traffic, head latency fractions, or speedup predictions.

## 4. Why an earlier hidden needs a different proof

Suppose an earlier hidden is h0 and the selected later hidden is h0+delta.
If a valid bound `||delta||_2 <= R_delta` exists, then:

`z_i(later) <= w_i^T h0 + b_i + ||w_i||_2*R_delta`.

A stronger pairwise test for candidate c is:

`(w_c-w_i)^T h0 + b_c-b_i > ||w_c-w_i||_2*R_delta` for every competitor i.

This is a valid sufficient condition in real arithmetic.
It does not provide R_delta.
An observed norm, a fitted predictor, or high cosine similarity is not a universal enclosure.
The remaining blocks include attention, nonlinear MLPs, residuals, normalizations, and depth-indexed KV reads.
This audit found no source-provided contraction guarantee for that finite trajectory.
Weight identity alone supplies none.

The CPU control uses h0=`[1,1023/1024]` and h1=`[1023/1024,1]`.
Both have equal norms.
Their cosine is about 0.9999995226969531.
The identity head selects token 0 for h0 and token 1 for h1.
A valid certificate for h0 still does not certify h1.
These vectors are not an Ouro trajectory.
The example rejects an inference from closeness alone, not all certified core bounds.

An earlier head can provide an incumbent for the final head.
That use does not require its earlier prediction to match the later winner.
The final bound must still cover every competitor at the final hidden.
Earlier projections add compute unless the baseline already needs them.
The same incumbent-ordering idea applies to ordinary non-recurrent models and prior tokens.

## 5. Original-gate threshold bound and state

Let c be the current exit CDF and s be survival before the current hazard.
Let `g=a^T h+b` be the current gate logit.
The current real-arithmetic exit condition is `c+s*sigmoid(g) >= q`, subject to the minimum depth.

For `s>0` and `0 < (q-c)/s < 1`, define `tau=(q-c)/s`.
Then the condition is equivalent to `g >= log(tau/(1-tau))`.
The partial-dot bound can enclose g exactly as it encloses a vocabulary row.
A lower bound above the threshold certifies exit.
An upper bound strictly below the threshold certifies continuation.
Boundary cases need explicit treatment instead of a logarithm outside its domain.

This gate has only 2048 weights and one bias.
Norm calculation, metadata, and launch costs can exceed its skipped dot-product tail.
Fixed R4 obtains no necessary gate-decision saving.
The current hidden still requires its full physical stack and loop norm.

There is also a state problem.
The next depth uses `c' = c+s*p` and `s' = s*(1-p)`, where `p=sigmoid(g)`.
A Boolean continuation result does not determine p.
Skipping the remainder of the gate does not preserve the exact policy state.
The minimum R2 setting does not permit omission of the R1 hazard from later CDF updates.

The control uses R1 hazards 1/10 and 2/5.
Both continue at R1.
With an R2 hazard of 1/5, their CDFs become 7/25 and 13/25.
At q=1/2, only the second exits.
Discarding the R1 hazard loses this distinction.

Conservative survival intervals could preserve exact decisions without exact internal gate values.
Such a method needs interval propagation, refinement, and fallback rules.
It is not impossible, but it is not the unchanged gate-state computation graph.
It also creates another generic interval-control algorithm to compare.
No such engine ran here.

## 6. Normalization and finite precision

For real arithmetic, the bias-free head has a useful generic identity.
If `h = D_gamma*x/sqrt(mean(x^2)+epsilon)`, its denominator is positive.
Therefore, `argmax_i w_i^T h = argmax_i w_i^T D_gamma*x`.
This can fuse the final norm and head for greedy selection.
It does not replace unfinished layers or earlier loop norms.
It also does not preserve ordinary sampling probabilities.

Actual BF16 casts can break this real-arithmetic identity's correspondence to the baseline implementation.
The same problem affects partial-head certificates.
If reference logit error is at most eta_i, enlarge each interval by eta_i.
Also include errors in prefixes, tail norms, and incumbent evaluation.
These error bounds must match the actual accumulation and cast path.
This audit did not derive an H100-specific kernel error enclosure.

The CPU counterexample uses weights `[[1,0],[1,1/512]]` and hidden `[1,1]`.
The real logits are 1 and 513/512.
A strict real certificate selects token 1.
Rounding these outputs to BF16 gives `[1,1]`.
The lowest-index tie rule then selects token 0.
The software cast is deterministic, but it is not an H100 GEMM measurement.

Greedy certification does not preserve a full categorical distribution.
The logits `[2,0]` and `[2,-100]` share a winner but have different softmax probabilities.
General sampling needs a separate exact sampling algorithm or a valid distribution-preserving verification step.
Arbitrary truncation of uncertified vocabulary rows is not such a step.

## 7. Generic nearest priors

| Primary source | Mechanism and relevant limit | Implication for C09 |
|---|---|---|
| [Ram and Gray, arXiv:1202.6101v1, 2012](https://arxiv.org/pdf/1202.6101) | Branch-and-bound exact MIPS uses geometric node bounds | Exact argmax pruning is an established generic family |
| [Li et al., FEXIPRO, SIGMOD 2017, section 2.2.1](https://lihui.info/doc/SIGMOD17.pdf) | Partial dot plus tail norm gives the same pruning bound | The core certificate is direct prior art |
| [Abuzaid et al., arXiv:1706.01449v3, 2019](https://www.bailis.org/papers/mips-icde2019.pdf) | Exact search must compete with blocked dense multiplication | Fewer scalar products do not establish lower latency |
| [Chen et al., arXiv:1810.12406v1, 2018](https://arxiv.org/pdf/1810.12406) | A trained screening model selects a reduced vocabulary subset | Ineligible as a new fitted predictor here, and not a universal full-vocabulary certificate |
| [FlashHead, author repository](https://github.com/embedl/flash-head) | The README describes clustered multi-probe retrieval and near-perfect recall | Treat lossless claims as unverified without an all-row bound or full verification |
| [Ruiz et al., FlashSampling, arXiv:2603.15854v3, 2026](https://arxiv.org/html/2603.15854v3) | Tiled projection fuses exact sampling and small distributed summaries | A strong generic control for logits materialization and communication, not tail-dot pruning |
| [Leviathan, Kalman, and Matias, ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html) | Target verification preserves the sampling distribution | A draft result alone is not the guarantee |
| [Zhang et al., Draft & Verify, arXiv:2309.08168v2, 2024](https://arxiv.org/html/2309.08168v2) | Training-free skipped-layer drafts receive full-model verification | No new training does not remove the verification cost |
| [Elhoushi et al., LayerSkip, ACL 2024](https://aclanthology.org/2024.acl-long.681.pdf) | Early-exit loss and layer dropout train useful early representations | Its training recipe is not a drop-in change to released Ouro weights |

The inspected FlashHead README is a source of mechanism descriptions, not a demonstrated exact certificate.
The OpenReview PDF request reached a browser challenge.
This audit did not establish its complete sampling guarantee.
The centroid control only proves that centroid ordering alone is insufficient for arbitrary clusters.
It does not evaluate the released FlashHead implementation.

The FlashSampling v3 HTML and unversioned abstract contain different performance wording.
This report transfers no performance number from either version.
Its distributional proof and floating-point equivalence remain distinct claims.
Its published end-to-end path uses sampling rather than this campaign's greedy mode.
The paper states that greedy mode disables that path.

## 8. CPU controls and evidence class

Run:

`python poc_ouro_parallelism_deep_discovery/scripts/certified_decision_cpu.py`

The script imports only the Python standard library.
It uses fixed rational tensors and explicit software BF16 output rounding.
All nine assertion groups passed.
Float64 dot products matched the exact rational values on these constructed cases.
It does not import Torch or read model tensors.
It does not write runtime state.
The saved output is [certified_decision_cpu.json](../artifacts/certified_decision_cpu.json).

| Control | Result | Scope |
|---|---|---|
| Strict partial-head certificate | Winner 0, structural projection count seven versus 16 dense | Harness computes full oracle dots and bound costs |
| Prefix winner reversal | The bound rejects a losing prefix-only winner | Fallback remains necessary |
| Exact tie | The strict test refuses certification | Preserve the baseline tie rule |
| BF16 output tie | A real certificate and software BF16 argmax disagree | Not a measured GPU kernel |
| Earlier/later hidden | Very close equal-norm vectors select different tokens | Not a model trajectory |
| Current affine gate | A partial dot certifies a zero-logit threshold | The current hidden is already ready |
| Gate survival state | Equal current actions produce different later exits | Boolean routing is not full policy state |
| Greedy versus sampling | Same winner, different probabilities | No distributional guarantee |
| Centroid ordering | Best centroid need not contain the best row | No full retrieval implementation test |

Evidence class: `CPU_ALGEBRA_CONTROL`.
Model quality: `NOT_RUN`.
GPU latency and achieved pruning on Ouro: `NOT_RUN`.
No toy operation-count ratio is a measured acceleration.

## 9. Both-model, both-mode, and P scope

| Scope | Final-head certificate | Earlier-hidden certificate |
|---|---|---|
| 1.4B fixed R4 | Algebra applies, model pruning and latency untested | No useful remaining-core enclosure established |
| 1.4B original adaptive | Algebra applies after output selection, gate state must match | Same gap, plus exact exit/KV policy |
| 2.6B fixed R4 | Same head dimensions, larger required core, untested | Same gap |
| 2.6B original adaptive | No new adaptive or quality test in this worker | Same gap |
| P2/P4 | Legal vocabulary sharding, communication costs unmeasured | No new legal concurrent core frontier |
| P8 | Projection only, no run or hardware authorization | No measured claim |

No head certificate changes historical or current KV ownership.
Skipping a recurrent pass would require a separate KV/state proof.
An earlier token hypothesis still needs exact verification and commit rules before it can become authoritative.

## 10. Falsification plan and verdict

First compare with graph-captured dense head plus fused/local argmax.
For a future scoped probe, predeclare m and use only non-fitted norm metadata.
Record all prefix, metadata, compaction, fallback, and communication costs.
Use unchanged selected hidden inputs from a mode-matched baseline.
Then test reference-kernel numerical agreement before any free-running quality claim.

Stop the recurrence-specific claim if aliased unrolling obtains the same benefit.
Stop kernel work if the optimistic skipped head work cannot cover added costs.
Reopen earlier-hidden certification only with a valid, cheap remaining-core enclosure.
That reopening must include KV versions and adaptive control.
A new similarity observation does not meet this requirement.

The final-head mechanism is legal in real arithmetic but generic.
Its actual useful headroom remains unknown.
This audit establishes no new recurrence-specific parallel axis.

## 11. Search and source ledger

All web access occurred on 2026-10-19 after 10:44:10 UTC.
Exact per-query timestamps were not captured.
Queries changed across five batches:

1. Exact vocabulary argmax, pruning, MIPS, and speculative verification.
2. Exact MIPS indexing, branch-and-bound, and vocabulary projection.
3. Exact LM-head pruning, Cauchy bounds, and fast vocabulary projection.
4. FlashHead, FlashSampling, and FEXIPRO.
5. Training-free Draft & Verify and trained LayerSkip.

The artifact preserves the full query strings.
Primary PDFs and official source supplied the technical claims above.
No aggregator supplied a final technical conclusion.
The local source map includes these anchors:

- Official `modeling_ouro.py`: per-loop norm/gate near lines 590–608, head and output selection near lines 630–806.
- CDB `adapters/ouro.py`: `recurrent_step`, `_exit_gate`, `exit_policy_spec`, and `lm_head`.
- CDB `model_runner.py`: coda projection and argmax near lines 1037–1105.
- CDB `exit_policy.py`: `should_exit` near lines 150–168.
- Installed vLLM `ouro.py`: recurrent forward and `compute_logits` near lines 350–441.
- Installed vLLM `logits_processor.py`: `_get_logits` and `get_top_tokens` near lines 138–211.

Official model source SHA256: `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`.
CDB adapter SHA256: `dc1eaaf92a4cec1e2c0ae14323674005f4f60ecea70fbd15fbaf385044b8c2ea`.
Installed logits processor SHA256: `b2b17669a923a57d34c0a0b94902c4e7742f88a73e57ca9b881c84207fceaded`.

FlashSampling author code revision: `6e376af5184d9d3a572755b5260c9c4368562828`, dated 2026-09-25.
This worker did not install or execute that source.
An initial FlashHead `main` branch lookup failed.
The repository metadata identifies `master` as its default branch.
The artifact records any resolved revision separately.

Concurrent worker time must not multiply campaign elapsed time.
This worker used no GPU, process control, installation, model-weight download, or git mutation.

The recorded end checkpoint is 2026-10-19 10:53:20 UTC.
The checkpoint span is 550 seconds.
The final CPU run occupied 2026-10-19T10:53:03.598135+00:00 through 2026-10-19T10:53:03.598800+00:00.
This duration is not a GPU or model latency result.
The report passed the prose test with zero findings before final ledger publication.

Script SHA256: `395d18d3814b2fc531a108be49e0a6099b210ba904171d1004656a7e93b4fb78`.

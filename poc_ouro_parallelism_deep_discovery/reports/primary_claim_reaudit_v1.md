# Primary-claim re-audit, version 1

Status: source audit only. Runtime reproduction is `NOT_RUN`.

## Outcome

The nearest-prior mechanisms remain relevant. Their results do not establish this campaign's complete comparison.
The audit found two inherited statements that need correction.
It also found several qualifiers that the final dossier must retain.
No candidate, benchmark, model, or disposition changed.

This report audits eight selected works.
The [structured matrix](../artifacts/primary_claim_reaudit_v1.json) records source hashes, pins, locators, and inspected extent.
A downloaded file hash does not certify a full-file read.

## Actual inspection depth

Each row names selected passages read during this segment.
It does not certify a full-paper or full-repository review.

| Work and pinned paper | Inspected paper passages | Inspected code |
|---|---|---|
| [LoopSpec, v1](https://arxiv.org/html/2609.17184v1) | Sections 4.1–4.3, Section 6 setup and Table 1, Appendix B, Appendix C opening | CLI restrictions, recurrent executor, rejection fallback, branch allocation and release |
| [WaveFront, v2](https://arxiv.org/html/2609.23033v2) | Algorithm 1 and Sections 3.2–4.2, Appendices B.4, C.2, and D | Dynamic scheduler and Ouro gate/cache helpers |
| [FlashLoop, v2](https://arxiv.org/html/2609.29812v2) | Sections 3.2–5.4, Appendices D.3, E, F protocol, and G | Configuration validation, BF16 cache selection, decode input restriction |
| [CDB, v2](https://arxiv.org/html/2608.09444v2) | Section 6, selected Appendix B latency discussion, Appendices C.1–C.2 and E.1–E.2 | Benchmark inputs, lengths, replay, coda argmax, token feedback, gate policy, terminal copies |
| [Enabling, v1](https://arxiv.org/html/2610.09013v1) | Section 3 mechanism, Section 7 limits, Appendices A.1–A.2, D.1, E.1–E.2 | No official implementation located in the inherited bounded search |
| [SNLP, v3](https://arxiv.org/html/2605.17842v3) | Section 3 correction/fusion passages, Section 4, Section 5 protocol, Table 3, Appendices E.3–E.4 | Batched correction and cached-generation KV update paths |
| [Quasi-DEER/ELK, v3](https://arxiv.org/html/2407.19115v3) | Sections 2–4 selected derivations, Proposition 1, Appendix A.1 corollary, Sections 6.1–6.3, Appendix B.1 implementation caveat | `seq1d` interface and quasi/trace controls |
| [KernelFlume, v1](https://arxiv.org/html/2606.29207v1) | Sections 3–5 selected ownership/protocol passages, Sections 6.1–6.5 | No official implementation located in the inherited bounded search |

## Corrections and retained limits

### 1. LoopSpec

Retain the direct overlap with mixed-depth self-speculation.
Its proposal-disagreement gate is not Ouro's checkpoint exit gate.
The inspected executor retains that checkpoint gate but never evaluates it.
The CLI rejects every parallel size other than one.
Branch management shares prefix indices, preserves private new KV, and requires ordered release.
The rejection code includes finite-precision fallback rules.
Its distribution argument does not certify native BF16 word identity.
The paper times decode after prefill on one Blackwell GPU.
Appendix C trains the DFlash comparator, not LoopSpec. [Paper](https://arxiv.org/html/2609.17184v1), [executor](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/modeling/ouro.py#L85), [CLI](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/server/cli.py#L88)

### 2. WaveFront

Retain the distinction between the main 2.42× generation-throughput result and the separate 2.30× graph timing.
The latter uses steady-state width and static shapes.
Appendix B.4 reports bounded FP32 token agreement, not a universal bitwise guarantee.
Cross-recurrence KV sharing is explicitly approximate.
Appendix D leaves adaptive-depth interaction and disaggregation for future work.
Public code contains adaptive and sampling paths.
Their existence does not validate the campaign's original-gate policy or distributed execution.
The inspected gate casts logits to FP32 before sigmoid.
Do not silently equate that operation order with the campaign gate. [Paper](https://arxiv.org/html/2609.23033v2), [gate helper](https://github.com/summerbro-hhj/wavefront-decoding/blob/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/ouro_blocks.py#L375)

### 3. FlashLoop

Retain “training-free, approximate.”
Task-score retention does not imply equal logits or tokens.
The main method reuses unselected attention contributions and inactive tokens.
It also stores INT4 KV residuals.
The BF16 cache switch removes quantization, not the two sparsity approximations.
The inspected engine requires batch-one, single-token decode.
Appendix E calibrates retention once with 128 WikiText-2 sequences and a 5% reconstruction-error limit.
Appendix G uses the engine with quantization and sparsity disabled as its efficiency baseline.
Do not describe its baseline as necessarily an unoptimized Hugging Face path. [Paper](https://arxiv.org/html/2609.29812v2), [configuration](https://github.com/Superone77/FlashLoop/blob/aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6/flashloop/config.py#L14), [engine](https://github.com/Superone77/FlashLoop/blob/aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6/flashloop/engine.py#L208)

### 4. CDB

Correct “replayed reference outputs” when it implies replay of token values.
The inspected benchmark uses seeded random prompts, reference response lengths, recorded exits, and no EOS termination.
The engine still computes LM-head argmax tokens and feeds generated tokens back.
Thus, the protocol measures actual model execution under controlled lengths and exits.
It does not validate live gates on natural free-running responses.

Also narrow “all experiments use only 1.4B and Huginn.”
Those models define the main serving comparison.
Appendix B/Figure 9 additionally includes Ouro-2.6B recurrent-step latency.

The original-head option exists.
Delayed transport can change its effective decision horizon.
Terminal copies precede deeper reads.
Synthetic replay pays staged signal readback but discards its value.
New lookahead/preloop training belongs to separate gate variants. [Paper](https://arxiv.org/html/2608.09444v2), [benchmark protocol](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/benchmarks/runner.py#L188), [real coda argmax](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/model_runner.py#L1036)

This correction applies to [the earlier dynamic-runtime report](/home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery/reports/dynamic_runtime_prior_boundary.md:31).
That report's reference-output wording is too broad.
Its active-gate limitation remains valid.

### 5. Enabling dynamic computation

Retain the inference-only original-gate path and the separate training track.
Terminal aliases preserve the stated last-exited value rule, not dense full-depth history.
Dense-Ragged keeps full-depth prefill.
ETE-Ragged changes prefill too.
Appendix A distinguishes physical page savings from ideal useful-FLOP savings.
Its nested runner retains unnecessary lane work.
Section E.2 omits non-thinking Ouro-2.6B ETE-Ragged results because of prefix instability.
Do not present that mode as validated across both Base sizes.
“H100 GPUs” does not establish a P2/P4 implementation.
Missing public code limits reproducibility, not mechanism priority. [Paper](https://arxiv.org/html/2610.09013v1)

### 6. SNLP

Retain the direct precedent for batched nonlinear forwards with structured depth corrections.
Do not state that every application requires new training.
Post-hoc variants exist.
Version 3 distinguishes biased finite-iteration inference from exact trace recovery.
Table 13's 0.64× and 0.63× results concern two selected off-the-shelf configurations, not an Ouro impossibility theorem.
Table 3 explicitly labels its verified-drafter end-to-end speedups theoretical.
The inspected generation path also recomputes KV from corrected inputs.
That projection work cannot disappear from an implementation cost account.
The selected paper passages do not establish BF16 precision or rank scope. [Paper](https://arxiv.org/html/2605.17842v3), [correction and KV code](https://github.com/phymhan/nanochat-snlp/blob/330ce33cd80d0cfcfdfdb58fe519574baf595777/snlp/inference_idn.py#L1008)

### 7. Quasi-DEER and ELK

Retain the finite-chain theorem with its assumptions.
Proposition 1 bounds undamped iterations by chain length in real arithmetic.
Its surrogate-matrix corollary does not promise fewer than four expensive rounds for Ouro R4.
Diagonal correction reduces the linear solve cost, not all nonlinear model work.
Appendix B.1 distinguishes specially derived GRU diagonals from dense-Jacobian extraction.
The latter can lose the stated memory benefit.
ELK adds stabilization machinery.
Section 6.3 reports a trained autoregressive GRU case where all parallel methods remain slower than sequential evaluation.
None of these facts proves universal failure on Ouro. [Paper](https://arxiv.org/html/2407.19115v3), [interface](https://github.com/lindermanlab/elk/blob/58782bd07a58c28b48c651ac9bd9ac2fec527d58/elk/algs/deer.py#L22)

### 8. KernelFlume

Do not expand “no Ouro C1–8 evidence” into “no low-batch evidence.”
Section 6.2 includes real A6000 scaling at batch sizes 2–64.
Section 6.3's pipeline experiment instead uses 16–128.
The real 8B tests and larger-model simulations remain separate.
Projection/FFN owners send queries and fresh KV to distinct attention owners.
Historical attention still waits for the exact query.
Partial returns, merge, route publication, and graph-external CPU signaling remain required work.
The filled-pipeline bound requires independent microbatches.
Trace replay does not establish the same protocol as CDB's recorded exits.
No inspected result closes the campaign's complete matrix. [Paper](https://arxiv.org/html/2606.29207v1)

## Evidence boundary

These sources establish direct mechanisms and scoped measurements.
They do not prove that every recurrence-specific physical ownership optimum is absent.
Generic graph expressibility is not a cost comparison.
Code restrictions describe the inspected revision, not mathematical impossibility.

The audit made no new literature search query.
It used direct pinned URLs, in-page finds, and targeted file reads.
All eight primary HTML bodies and the selected pinned code files were retrievable.
Two earlier official-code searches remain unresolved.
This segment did not repeat them or convert them into proof of absence.
The matrix records local path-resolution mistakes separately from external retrieval failures.

No GPU job, model execution, quality test, or held-out evaluation occurred.
The disjoint baseline extension does not change these candidate-source observations.

## Time

Work started at 2026-10-19T17:27:57Z.
Source inspection ended at 2026-10-19T17:34:54Z.
That interval is 417 seconds.
Report finalization and validation reached the accounting cutoff at 2026-10-19T17:37:13Z.
The full recorded segment is 556 seconds, or 9 minutes 16 seconds.
The ledger excludes final timestamp stamping after that cutoff.
These minutes overlap coordinator wall time.
They must not be added to campaign elapsed time.

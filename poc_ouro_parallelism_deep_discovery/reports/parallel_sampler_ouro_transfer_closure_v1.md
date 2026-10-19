# Recurrent-depth sampler: unchanged-Ouro transfer closure v1

## Decision and scope

Decision: `NO_TRANSFER_ESTABLISHED_UNCHANGED_TARGET`. This audit proposes zero new candidates.

The source supplies a training-free sampler for its compatible pretrained model. It does not supply an exact transformation for unchanged Base Ouro. The missing input interface is a structural difference. It is not evidence that every alternative sampler is impossible.

This report closes one concrete lead. It extends [the earlier sampler audit](parallel_sampling_nearest_audit.md). It does not count the paper as a new discovery. Model execution, GPU tests, latency measurements, and Ouro quality tests are `NOT_RUN`.

## Identity, version, and publication limit

The authors are Jonas Geiping, Xinyu Yang, and Guinan Su. The title is *Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models*. arXiv lists only v1, submitted 2025-10-16 at 17:59:07 UTC. Its code link resolves to `seal-rg/recurrent-pretraining`. [arXiv record](https://arxiv.org/abs/2510.14961)

A later ICML 2026 record exists at PMLR 306:34442–34462. The inspected reader rejects its PDF as `application/octet-stream`. The technical claims below therefore pin arXiv v1, not an assumed identical final body. Status: `FINAL_CAMERA_READY_BODY_UNAUDITED`. [Publication record](https://proceedings.mlr.press/v306/geiping26a.html), [final PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/geiping26a/geiping26a.pdf)

## What training-free requires

Sections 3.1–3.2 require separately injected token conditioning and useful intermediate readouts. Random initialization is optional. Cross-recurrence KV sharing is beneficial, not mandatory. These are properties of the pretrained model, not components that inference can add without changing it. The study uses Huginn-0125. A separate MetaMath model receives finetuning. [Pinned paper, Sections 3.1–3.2 and 5.1](https://arxiv.org/html/2510.14961v1#S3.SS2)

Remark 3.1 assumes contraction for convergence toward the autoregressive solution. It does not establish finite-R4 exactness. Section 5 uses batch one, Hugging Face dynamic caches, CUDA events, and sandboxed A100-40GB devices. Its task-score tradeoff is not an exact-output guarantee. [Pinned paper, Remark 3.1 and Section 5](https://arxiv.org/html/2510.14961v1#S3.SS5)

## Official code and target comparison

The live repository HEAD remains `1ea7220ec7eb42d13e89db0663df254d0bcdc28e`. Its commit timestamp is 2025-12-29T10:31:25Z. These code claims refer only to the inspected files and functions.

| Required operation | Inspected official sampler | Unchanged Base Ouro |
|---|---|---|
| Inject current token conditioning | `raven_modeling_minimal.py:554` constructs a learned `2H → H` adapter. Line 776 applies it to concatenated state and conditioning at each recurrence. | Both pinned files embed once at lines 535–536. Lines 592–608 pass the hidden state through successive loops. They do not reinject token embeddings. |
| Revise drafts while retaining latents | Lines 1335–1380 recompute conditioning, optionally mix embeddings or noise, update latents, and replace draft tokens. | Retaining a state from a changed token or prefix does not certify the new token's recurrent trajectory. |
| Read and freeze | Lines 1361–1366 read intermediate latents. Lines 1396–1443 use latent-change criteria. No target verification appears in this inspected generator. | Lines 606–608 apply the original gate after loop normalization. Lines 722–733 construct exit masses. A latent-change threshold is a different rule. |
| Address historical KV | Lines 169–219 compress or select recent states. The default at line 1242 is `latest-m4-compress-s4`. | Line 320 addresses `current_ut * num_hidden_layers + layer_idx`. Computed depths have separate historical KV. |

Source links: [Huginn adapter](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/raven_modeling_minimal.py#L554), [generator](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/raven_modeling_minimal.py#L1235), [cache](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/raven_modeling_minimal.py#L148), [Base 1.4B](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py#L592), [Base 2.6B](https://huggingface.co/ByteDance/Ouro-2.6B/blob/1ed04250da1a9936042725d302e81c8fa2ab5abd/modeling_ouro.py#L592).

The code's modulo four selects Huginn's physical core blocks. It does not identify Ouro's four recurrent depths. The generator asserts batch size one at line 1262. Its default noise differs from the paper's main setting. These details require pinned arguments, not assumptions about equivalence.

The repository also contains a vLLM plugin. Its README explicitly leaves token-level adaptive execution unimplemented at line 29. Its inspected forward runs fixed `mean_recurrence` at lines 396–410. This is not evidence of a deployed diffusion-wavefront engine. The README permits ordinary data-parallel evaluation. [Plugin scope](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/vllm/README.md#L15), [fixed forward](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/vllm/raven_vllm.py#L375)

The earlier audit records a stopping-rule discrepancy between Equation 3 and the code. This closure does not treat that discrepancy as a tested defect. Neither version supplies an Ouro exactness guarantee.

## Transfer screen

The following reasoning is `ANALYTIC_SCREEN`. It is not a measured result.

Let `H_k(x, K)` denote the authoritative state after k Ouro passes. Here `K` includes every depth-specific historical cache used by those passes. A retained draft state is `H_k(x_old, K_old)`. An exact reuse transformation would need:

```text
U_k(H_k(x_old, K_old), x_old, x_new, K_old, K_new)
    = H_k(x_new, K_new).
```

It must also recover every authoritative internal KV row and original gate input. The inspected source supplies no such transformation for Ouro. A cheap transformation is not implied by tied weights. Updated attention context can influence later states. That fact alone does not undo earlier computation under a different token or prefix.

| Proposed transfer | Decision under unchanged weights, BF16, and original gate |
|---|---|
| Add the Huginn adapter or reinject embeddings | Changes the recurrence. The existing Ouro checkpoint has no matching trained adapter in this execution path. Reject within current scope. |
| Keep all depth-specific KV and disable Huginn sharing | Avoids one approximation. It does not supply the missing state transformation or preserve draft-dependent internal KV. No independent candidate. |
| Reuse partial states after a draft changes | Requires the equation above, exact repair, or authoritative recomputation. Unverified stale-state reuse repeats the rejected assumption. |
| Substitute latent stability for the original exit gate | Changes the adaptive policy. Useful intermediate logits do not establish interchangeable gate decisions. Reject within current scope. |
| Verify and repair provisional branches | Returns to the existing C07 branch-verification question. Exact correction returns to C05. This source adds no distinct exact shortcut. |
| Batch only independent, ready work | Legal in principle. Generic mixed-depth batching remains the matched baseline, not a new mechanism from this paper. |

The source's serial recovery settings concern its own sampler. They do not prove that a useful multi-token Ouro wave preserves the fixed-R4 result or original adaptive trajectory.

Ouro does expose intermediate readouts. This audit does not claim that they are unusable. Their existence does not prove the paper's changing-conditioning prerequisite. No new quality failure or quality success follows from this source inspection.

## Adaptive KV boundary

The campaign selects the original checkpoint gate, q=0.5, minimum R2, maximum R4, and `last_exited`. It disables delayed consumption, asynchronous batching, and synthetic exit replay. The loader checks gate tensor identity. This audit reads those checks but does not execute them. [Campaign loader](../runtime.py#L28)

CDB applies the original gate after normalization. Its exit-copy code fills only the missing deeper KV rows of an already exited token. It does not replace every computed depth with the latest recurrence. Its implementation assumes at most one recurrent token per request in flight. A revised branch protocol would need explicit ownership and repair. [Pinned gate adapter](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/adapters/ouro.py#L249), [exit-copy protocol](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/continuous_api.py#L1190)

## Reopening condition and limits

Reopen only with a distinct exact transformation, a valid certificate, or a separately authorized target change. An Ouro transformation must preserve all required states, depth KV, and original gate decisions. It must reduce charged work against ordinary TP of the same ready frontier. This source does not provide that evidence.

Source records, retrieval reasons, and actual time records are separate artifacts. The audit changes no earlier source ledger, candidate card, runtime, model, or raw output.

## Validation and time account

Both JSONL files parse. Their identifiers are unique. The decision JSON parses and records zero candidates. Every recorded SHA256 field has 64 hexadecimal characters. The report linter reports zero findings. These are document checks, not model tests.

The worker segment starts at 2026-10-19T15:30:50Z. Its accounting cutoff is 2026-10-19T15:37:34Z. The observed interval is 404 seconds, or 6 minutes 44 seconds. Final record formatting follows that cutoff. This concurrent interval is not additional campaign elapsed time. It contains no deliberate wait, GPU execution, model execution, or package installation.

Artifacts: [sources](../artifacts/parallel_sampler_ouro_transfer_sources_v1.jsonl), [retrieval log](../artifacts/parallel_sampler_ouro_transfer_searches_v1.jsonl), [decision](../artifacts/parallel_sampler_ouro_transfer_decision_v1.json), [time account](../artifacts/parallel_sampler_ouro_transfer_time_v1.jsonl).

# CacheFlow and C18 source-completeness supplement v1

## Outcome

CacheFlow is now a method-read restoration prior, not only an abstract-level lead. Its exact official implementation identity remains unavailable in the inspected primary links and current records. This supplement narrows the corresponding gap in the [final novelty audit](final_novelty_claim_audit_v1.md). It changes no candidate disposition.

OpenXLA's pinned CSE implementation documents static duplicate elimination. It does not implement C18's complete runtime equality and state-publication protocol.

## CacheFlow: source and read extent

The paper is [arXiv:2604.25080v2](https://arxiv.org/abs/2604.25080v2), revised 2026-09-26. I read Sections 3.1–3.2, Section 4, the reproducibility statement, and Appendix A's assumptions and restoration bounds. This includes A.2's device mechanism. I did not independently reproduce experiments or formally check every proof.

The method recomputes leading tokens while loading later KV. Layer splits generalize into monotone staircase boundaries. Saved shard-boundary activations let devices restore known history independently. Its planner balances recomputation against loading, including batch contention. [Method and Equations 1–5](https://arxiv.org/html/2604.25080v2#S3)

Appendix A assumes identical blocks, additive exact costs, sufficient buffers, and interference-free compute/load overlap. Its guarantee excludes estimated costs and non-additive batched costs. The device bound ignores suffix-activation transfers. These are model restrictions, not measured runtime failures. [Appendix A](https://arxiv.org/html/2604.25080v2#A1)

Section 4 reports A100/H100 experiments on Llama/Qwen with vLLM/LMCache, including multi-GPU and TP/PP settings. It does not test Ouro. The reproducibility statement promises code upon acceptance. No official repository link or revision appears in the inspected primary record. Code-level numerical and ownership checks therefore remain unavailable here. This is not proof that code exists nowhere. [Evaluation and reproducibility](https://arxiv.org/html/2604.25080v2#S4)

## Transfer boundary for C10

The following statements are transfer deductions.

Saved boundary activations remove a restoration dependency because that historical input already exists. They do not supply an uncomputed current-token hidden state. An Ouro adaptation must retain the correct request, depth, layer, position and checkpoint versions. It must charge every saved activation, retrieval, reconstruction, current-KV append and publication.

The paper's recurrent-layer states belong to hybrid sequence architectures. They are not evidence that Ouro's depth-indexed caches can merge. Native stationary KV also supplies no external restoration transfer to remove. Thus CacheFlow strengthens generic C10 overlap, but supplies no measured unchanged-Ouro advantage or ready executable control. No BF16 parity or original-gate guarantee follows.

## Static CSE versus dynamic C18 equality

I pinned OpenXLA at `e646c974796caf1f134393934236eb91888ab69f`, dated 2026-10-19 17:46:36 UTC. The [pipeline document, lines 106–108](https://github.com/openxla/xla/blob/e646c974796caf1f134393934236eb91888ab69f/docs/hlo_to_thunks.md#L106) places HloCSE after multi-output fusion.

The actual [header, lines 32–35](https://github.com/openxla/xla/blob/e646c974796caf1f134393934236eb91888ab69f/xla/service/hlo_cse.h#L32) specifies identical instructions with the same operands. The [key, lines 139–197](https://github.com/openxla/xla/blob/e646c974796caf1f134393934236eb91888ab69f/xla/service/hlo_cse.cc#L139) uses IR operand identities, with explicit iota/buffer exceptions. [RunOnComputation, lines 339–395](https://github.com/openxla/xla/blob/e646c974796caf1f134393934236eb91888ab69f/xla/service/hlo_cse.cc#L339) checks instruction equivalence and safe removal before replacing uses. [Lines 317–319](https://github.com/openxla/xla/blob/e646c974796caf1f134393934236eb91888ab69f/xla/service/hlo_cse.cc#L317) exclude side-effecting instructions.

CSE does not compare runtime BF16 tensors from different recurrent invocations. Shared W in `F(W,x_r)` and `F(W,x_s)` does not make their different operand definitions equal. Proven identical immutable operands can support static elimination. Equality that appears only at runtime needs another protocol.

C18 additionally requires:

1. Exact equality or safe immutable identity for the complete read set, not only a hash match.
2. Matching dtype, layout, positions, model parameters and numerical execution contract.
3. Matching historical KV contents or provenance for attention.
4. Valid output lifetime and every required write, publication, gate update and sampling action.
5. Collision handling, invalidation, reader completion and charged lookup/transfer costs.

Identical hidden words with different histories cannot authorize attention reuse. Identical MLP inputs can justify a narrower pure-operation reuse test, but not omission of the original hazard/CDF updates. These are C18 obligations, not an implemented extension of HloCSE.

The generic memoization comparison remains valid. The stronger claim that static CSE already supplies a complete dynamic C18 coalescer remains unsupported.

## Records and scope

The [source record](../artifacts/cacheflow_cse_source_supplement_v1.json) preserves hashes, exact read extents, retrieval limits and actual UTC time. It distinguishes rendered paper locators from raw code line numbers. One incorrect CSE source path returned 404. The correct service path resolved.

No keyword search, GPU/model execution, installation, new card, protected-outcome access, architecture promotion, or old-file edit occurred.

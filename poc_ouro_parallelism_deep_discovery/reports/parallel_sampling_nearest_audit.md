# Parallel sampler and finite-correction audit

Status: targeted source audit complete. Ouro runtime, GPU timing, and task-quality tests are `NOT_RUN`.

## Outcome

A finite-chain exact-correction candidate is not new as a general algorithm. Jacobi, Newton, arbitrary finite surrogate matrices, triangular Anderson safeguards, and structured layer corrections already have close precedents. The open question is narrower: can an inference-only method recover authoritative Ouro states and KV in fewer exposed core passes than the strongest matched baseline?

No source inspected here establishes that result for Ouro at four recurrent passes. A finite upper bound of four correction rounds is not a useful strict reduction from four sequential passes. Small residuals, stable tokens, and task-score parity are different guarantees.

This audit extends the eight-work audit in [literature_nearest_audit.md](literature_nearest_audit.md). The source registry is [parallel_sampling_sources.jsonl](../artifacts/parallel_sampling_sources.jsonl). It records search queries, versions, official-code revisions, access times, and inspection limits. Missing code or unreadable content is not evidence that a claim is false.

## P01: the recurrent-depth sampler

The exact title is *Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models*. arXiv lists v1, dated 2025-10-16. The paper targets Huginn-0125, a 3.5B P/R/C model with default recurrence 32. It does not test Ouro. Repeated input injection permits course correction as token conditioning changes. Intermediate states must support readout. Cross-recurrence KV sharing is an empirically tolerated Huginn property, not an Ouro guarantee.

The main sampler needs no new training. The authors finetuned a separate MetaMath variant for one epoch. Main latency tests use one sandboxed A100-40GB, batch one, CUDA events, and median per-query tokens/s. The default inner recurrence is four. Remark 3.1 assumes a contractive recurrent operator for convergence toward the autoregressive solution. The paper does not establish contractivity for the deployed model. It gives no exact finite-R4 or distribution-preserving verification guarantee. Remark 3.2 charges repeated P/C work. Appendix A.2 describes possible state-slot engines and device movement, but does not implement them. [P01, Sections 3–5 and Appendix A](https://arxiv.org/html/2510.14961v1)

A newer [ICML 2026 publication record](https://proceedings.mlr.press/v306/geiping26a.html) exists: PMLR 306:34442–34462. Its final PDF body is `FINAL_CAMERA_READY_BODY_UNAUDITED`. The current reader rejected its content type. OpenReview returned a browser challenge. No installed PDF parser was available. The worker installed no dependency. Thus the complete technical audit pins arXiv v1, not an assumed unchanged camera-ready text.

### Official code

The observed revision is `1ea7220ec7eb42d13e89db0663df254d0bcdc28e`. In `recpre/raven_modeling_minimal.py`, `core_block_forward` concatenates the recurrent state and token embedding through the trained adapter, lines 764–781. `generate_diffusion_style`, lines 1235–1549, asserts batch size one. It evolves latent states, mixes token embeddings, samples new tokens, and freezes a prefix. Its default cache lookup is `latest-m4-compress-s4`, not depth-separated Ouro KV. The code's default noise value differs from the paper's main default. A reproduction must pin arguments. [Pinned model code](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/raven_modeling_minimal.py#L1235)

There is an inspected specification difference. Equation 3 requires all preceding latent-change tests to pass. Lines 1437–1443 locate the last passing position and freeze through it without an explicit contiguous-prefix test. This is not a runtime-tested defect. It is a reason to test the actual stopping rule before importing a guarantee. [Pinned stopping code](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/raven_modeling_minimal.py#L1437)

## Nearest forward and backward sources

All values and guarantees in this table are `SOURCE_REPORTED`. None is an Ouro measurement. Official-code identities and audit depth are in the registry. A code identity check is not a code correctness audit.

| ID | Primary source and pinned version | Relevant mechanism and limit |
|---|---|---|
| P02 | [Accelerating Feedforward Computation via Parallel Nonlinear Equation Solving, 2002.03629v2](https://arxiv.org/html/2002.03629v2) | Section 3.2, Proposition 1 proves exact Jacobi recovery in at most T rounds for a triangular chain with zero tolerance. Hybrid methods follow. Section 4.1 gives a strict Markov chain that needs T rounds. Idealized processor cost omits movement and synchronization. Identical weights alone do not shorten the bound. |
| P03 | [Parallelizing non-linear sequential models over the sequence length, 2309.12252v3](https://arxiv.org/html/2309.12252v3) | DEER uses Newton updates and an associative scan for the linearized recurrence. Local quadratic convergence does not imply few rounds from an arbitrary guess. Dense D-dimensional state requires O(TD²) storage and O(TD³) scan work. V100 experiments concern long sequences and small RNN states, not four Ouro passes. |
| P04 | [Towards Scalable and Stable Parallelization of Nonlinear RNNs, 2407.19115v3](https://arxiv.org/html/2407.19115v3) | Section 3.2, Proposition 1 proves at most T undamped Newton iterations for any initialization, in exact arithmetic. Its corollary permits arbitrary finite surrogate matrices with the same bound. Quasi-DEER uses diagonal approximations. ELK adds damping and stability costs. Some regimes remain slower than sequential execution. |
| P05 | [Predictability Enables Parallelization of Nonlinear State Space Models, 2508.16817v4](https://arxiv.org/html/2508.16817v4) | Equation 10 and Theorems 2, 4, and 5 tie conditioning and convergence to finite-time Jacobian products and a convergence basin. Transient factors matter. High cosine similarity or small state change does not establish these hypotheses. Long-sequence asymptotics do not establish R4 headroom. |
| P06 | [DeepPCR: Parallelizing Sequential Operations in Neural Networks, 2309.16318v2](https://arxiv.org/html/2309.16318v2) | Newton solves a block-bidiagonal nonlinear trace through parallel cyclic reduction. Sequential reduction stages are O(cN log L), with nonlinear iteration count cN. Matrix work and memory remain. Large-depth experiments do not make four-step correction free. Official code was not located in this bounded search. |
| P07 | [Parallel Sampling of Diffusion Models, 2305.16317v3](https://arxiv.org/html/2305.16317v3) | ParaDiGMS fixes noise, applies parallel drifts and prefix sums, then advances a sliding window by a residual criterion. Proposition 2 bounds distribution error under stated assumptions. Positive tolerance is not exact equality. Image experiments use up to eight A100s. Twenty-five- and thousand-step evidence does not establish R4 amortization. |
| P08 | [Accelerating Parallel Sampling of Diffusion Models, 2402.09970v2](https://arxiv.org/html/2402.09970v2) | ParaTAA adds triangular Anderson history and a regularized small solve. Theorem 3.6 adds a safeguard to preserve finite triangular convergence. Ordinary Anderson acceleration alone lacks that bound. Positive-tolerance or fixed-iteration stops remain approximate. The official README describes eight-GPU deployment. |
| P09 | [SNLP: Layer-Parallel Inference via Structured Newton Corrections, 2605.17842v3](https://arxiv.org/html/2605.17842v3) | Direct depth-axis precedent: batched nonlinear layer forwards plus identity, diagonal, or structured correction. Version 3 states that useful gains come from biased finite-iteration inference, not exact trace recovery. Main gains use training regularization or finetuning. H100 tests cover Nanochat models. Appendix E.4 reports 0.64× and 0.63× at matched perplexity for two off-the-shelf models. These are not Ouro impossibility results. |
| P10 | [Accelerating Transformer Inference for Translation via Parallel Decoding, 2305.10427v1](https://arxiv.org/html/2305.10427v1) | Unmodified translation models use Jacobi and Gauss-Seidel hybrids. Full token-vector equality supplies a greedy fixed-point stop with an at-most-m iteration bound. This is not a general stochastic-distribution guarantee or a depth-axis method. Fewer iterations did not always reduce latency. |

### The strongest direct refutation: SNLP

The inspected SNLP code already implements the basic correction pattern. `forward_idn_batched`, lines 494–530, stacks old states, performs batched nonlinear forwards, and applies `_idn_correction`, lines 373–391. The correction uses a cheap identity-surrogate recurrence. Chunked variants also exist. Naming an identity correction a new recurrence-axis solver would not distinguish it from this source. [Pinned correction implementation](https://github.com/phymhan/nanochat-snlp/blob/330ce33cd80d0cfcfdfdb58fe519574baf595777/snlp/inference_idn.py#L373)

Autoregressive code applies a correction and then calls `_update_chunk_kv_cache`, lines 1008–1021. The KV path recomputes normalization, K/V projections, and RoPE from corrected inputs, lines 1064–1108. Cache repair is existing machinery and a real cost. Correcting a whole-core output does not certify the internal-layer KV produced from a wrong input. An Ouro design must repair that state or prove that its proposed representation preserves it. [Pinned cache update](https://github.com/phymhan/nanochat-snlp/blob/330ce33cd80d0cfcfdfdb58fe519574baf595777/snlp/inference_idn.py#L1008)

## Finite-R4 correctness screen

This section is `ANALYTIC_BOUND`, not measured speedup or Ouro convergence evidence.

For a fixed token and immutable historical cache snapshot, write:

```text
h_r = F_r(h_(r-1); historical_KV_r),  r = 1,...,4.
```

The weights can be tied while `F_r` differs through depth-specific attention context. Each speculative evaluation must write current-token KV to isolated scratch state. Otherwise evaluation changes the residual operator, and the pure triangular-chain argument no longer applies.

Jacobi makes at least one additional prefix state exact per round in exact arithmetic. It therefore gives an at-most-four-round guarantee here. One exact initial pass leaves at most three generic rounds. Neither fact guarantees a strict latency reduction. Newton and cheap surrogate corrections may propagate more information per round. Their finite-chain guarantee already appears in P04. A useful result requires earlier recovery on this model, with all correction and validation costs included.

A zero residual is a mathematical state-consistency condition. A nonzero tolerance is approximate. Exact mathematical recovery also does not prove bitwise BF16 equality after a different batching or reduction order. The campaign must state whether it requires reference logits, greedy tokens, a sampling distribution, or a bounded task-quality change. These cannot share one label.

### Internal KV is part of the result

Correct boundary states are insufficient if speculative evaluations cached K/V at incorrect internal inputs. Possible resolutions include a final batched evaluation from corrected inputs or selective internal repair. Another option is a proof that the maintained state already includes every authoritative internal value. Each resolution has a cost. A shallow-state cache must not become authoritative merely because a final boundary passes a check.

The CPU probe below verified two scalar identities. It did not run a model. For `F_r(x)=x²+r`, zero-initialized Jacobi needs four rounds to recover the full trace. For `F_r(x)=2x+r`, one exact Newton step recovers all boundary states. However, the inputs used by its initial speculative evaluations remain wrong at later depths. A separate final evaluation uses the corrected inputs. Full output is in the registry.

```python
# CPU-only reproduction; standard library only.
h0, depth = 1, 4
def trace(fn):
    out = []
    for r in range(1, depth + 1):
        out.append(fn(h0 if r == 1 else out[-1], r))
    return out
nonlinear = lambda x, r: x*x + r
ref, guess = trace(nonlinear), [0]*depth
for k in range(1, depth + 1):
    guess = [nonlinear(h0 if r == 1 else guess[r-2], r)
             for r in range(1, depth + 1)]
    assert guess[:k] == ref[:k]
assert ref == [2, 6, 39, 1525]
old = [0]*depth
residual = [old[r-1] - (2*(h0 if r == 1 else old[r-2]) + r)
            for r in range(1, depth + 1)]
delta = []
for i, value in enumerate(residual):
    delta.append(-value + (0 if i == 0 else 2*delta[-1]))
corrected = [old[i] + delta[i] for i in range(depth)]
assert corrected == [3, 8, 19, 42]
speculative_inputs = [h0] + old[:-1]
authoritative_inputs = [h0] + corrected[:-1]
assert speculative_inputs != authoritative_inputs
assert [2*x+r for r, x in enumerate(authoritative_inputs, 1)] == corrected
```

## Finite-R4 cost screen

Use exposed critical-path time, not a count of nominal stages:

```text
T_candidate = T_initial_guess
            + sum(T_batched_core + T_correction + T_communication + T_sync)
            + T_validation + T_KV_finalization + T_commit.
```

Charge only exposed portions of overlapping work. Record the batch shape and KV layout for every core call. Do not assume that one batched round costs one serial pass, or that P ranks divide this cost by P. Shared-weight batching can change compute efficiency. Communication, extra P/C readouts, scratch allocation, and cache repair can erase the gain.

For illustration only, suppose each correction round and one full finalization each cost one normalized pass. Ignore every other cost. Zero initial passes, two correction rounds, and one finalization cost three units against four. One initial pass plus those operations costs four. Three correction rounds plus finalization also cost four. Four correction rounds plus finalization cost five. These are structural accounting examples, not lower bounds, modeled speedups, or measured results. Some designs can reuse finalization work. That reuse needs evidence.

The comparison must include same-P TP applied to LoopSpec or WaveFront, as well as the strongest matched sequential serving path. A weaker four-pass baseline cannot establish a recurrence-specific systems advantage.

Adaptive decoding has a smaller budget when the original gate exits at depth two or three. The solver must preserve the original gate, threshold, minimum depth, and skipped-depth KV policy. Computing four depths before deciding to skip does not provide the original physical skip. A surrogate gate or changed cache policy changes the experiment.

## Changed-assumption tests

These tests distinguish a new mechanism from a renamed rejected assumption. They do not pre-authorize GPU work.

| Changed assumption | What can differ from a prior failure | Required evidence and stop condition |
|---|---|---|
| Stale guesses become provisional, not authoritative | An exact correction and validation path can remove the earlier quality mechanism. | Check state residuals, logits, and all KV slots. Stop if validation or repair consumes the saved passes. Generic correction is already P02/P04/P09. |
| A cheap untrained surrogate replaces a full Jacobian | Identity or diagonal operators reduce solver cost. | Compare directly with SNLP and quasi-DEER. Show earlier authoritative recovery, not only smaller norms. A fitted predictor changes the no-new-training scope. |
| A causal initial guess replaces an oracle | Previous-token or shallow-current-token state can initialize a solver. | Count the initial work and test held-out prompts. No future exact states may enter initialization. Stop if round tails remove R4 savings. |
| Shared weights change distributed residency | Mixed-depth batches may reuse local tied weights. | Compare against aliased unrolling and ordinary TP of the same speculative batch. Include remote KV, migration, pruning, and commits. Generic state ownership alone is not a recurrence-specific contribution. |
| Exact adaptive exits replace fixed-depth completion | An early authoritative prefix can preserve a physical gate exit. | Match gate decisions and skipped-depth KV on both Base sizes. Stop on unexplained policy mismatches or negative end-to-end latency. |
| Predictability reduces correction rounds | Favorable finite-depth dynamics may permit early recovery. | Measure finite-depth conditioning or a valid certificate. Cosine similarity and long-chain theorems are insufficient. No inspected source supplies this evidence for Ouro. |

## Remaining uncertainty and time account

The finite-chain idea remains mathematically possible. Close primary sources refute its generic novelty. Its useful R4 cost and model-specific correctness remain untested. None of this audit's CPU identities is evidence of Ouro contraction or quality.

The recorded worker segment ran from 2026-10-19T10:06:50Z to 2026-10-19T10:19:17Z. Its observed duration is 747 seconds, or 12 minutes 27 seconds. This is the accounting cutoff for this report. The segment overlaps the coordinator's elapsed time. It must not be added to campaign elapsed time as another worker's hours. The segment includes source reading, code inspection, two scalar CPU checks, writing, and validation. It contains no deliberate wait.


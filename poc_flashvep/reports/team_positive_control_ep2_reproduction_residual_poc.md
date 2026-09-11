# TEAM positive-control → true EP2 → residual opportunity PoC

## Executive decision

**Final label: `POSITIVE-CONTROL-ONLY`.**

The known positive result is real in this environment. Using the official TEAM
and SDAR modeling/generation code on the same SDAR-30B-A3B checkpoint, TEAM
achieved a median **1.832x** clean speedup over three bounded single-H100 restart
pairs. The benefit survived semantics-preserving true EP2: the reference EP2
path delivered a **1.289x median** across three restart pairs and **1.551x** at
the warm paired point.

The successor search did not clear the research gate. A production-like fused
expert control removed **56.94%** of TEAM reference-path latency using an
already-available vLLM primitive, shrinking TEAM's residual speedup to 1.179x.
After that control, the best independent perfect oracle was only **6.28% request
E2E** and had no demonstrated feasible or novel mechanism. EP4 follow-up is not
justified by these data.

## Research contract and scope

The investigation followed the required order:

`official positive control → true EP2 → residual profiling → candidate oracle`

No new optimization was inserted into Stage A. Stage B preserved TEAM's
decoded-token cache, hot/cold classification, speculative exploration, expert
restriction, selected experts, and decoder policy. Only physical expert
ownership, remote dispatch, local execution, and combine changed.

The official repository reports 1.94x average and up to 2.2x HumanEval
speedup.^1 The local result is not presented as a reproduction of that exact
leaderboard number: it is a bounded positive-control covering GSM8K and
HumanEval, followed by a mechanism-focused true-EP2 extension.

## Environment and fidelity

| Item | Value |
|---|---|
| Model | `JetLM/SDAR-30B-A3B-Chat-b32` |
| Checkpoint revision | `c351bbc37d240aa6871f167e8f92d694281b0c22` |
| TEAM commit | `e9c502e5753ce79f660371e2fb4a8666f66cae75` |
| SDAR commit | `4c2749ba103448f45520e8411533710a1e66574d` |
| Model structure | 48 layers, 128 experts, top-8, hidden 2048, expert intermediate 768 |
| Stage A | physical GPU 6, one H100 80GB, FP16 |
| Stage B/C | physical GPUs 6 and 7, TP1/DP1/EP2 |
| Software | Python 3.11.15, PyTorch 2.8.0+cu128, NCCL 2.27.3, Transformers 4.52.4 |

The released OpenCompass tree imports many unrelated optional backends and its
environment bundle is incomplete in the isolated setup. Rather than modify the
algorithm, the harness loads the released generation function bodies directly
from the official wrapper. Checkpoint, prompts, block length 32, 32 denoising
steps, threshold 0.95, greedy top-k 1, dtype, and modeling files remain matched.

## Stage A — official TEAM trend

### Clean latency

| Restart | Sample count | Baseline mean | TEAM mean | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 8 | 58.251 s | 24.949 s | 2.335x |
| 2 | 2 | 59.233 s | 34.565 s | 1.714x |
| 3 | 2 | 59.406 s | 32.429 s | 1.832x |
| **Median speedup** | — | — | — | **1.832x** |

### Work direction

The 32-token structural trace found NFE and MoE calls fell from 24/1,152 to
14/672 (-41.7%). Active-expert events fell from 63,280 to 31,134 (-50.8%).
This agrees with TEAM's stated goal of using temporal/spatial consistency to
activate fewer experts for useful accepted progress.^2

Token-expert pairs increased 11.5% because TEAM evaluates four speculative
candidates in larger forwards. This prevents an overly simple “TEAM removes
all MoE work” interpretation: it removes forward/activation events while
trading some of them for wider batched speculation.

### Quality

At 128 output tokens, GSM8K matched at 3/4. HumanEval was 2/4 baseline versus
1/4 TEAM, but the differing TEAM case ended mid-function. Repeating that case
and the truncated GSM8K case at 256 tokens yielded 2/2 for both methods. The
subset supports the positive-control gate, not a full benchmark-quality claim.

Instrumentation added 36.7% to the baseline and 12.2% to TEAM at the traced
point. All headline latency results therefore use clean runs.

## Stage B — true EP2

### Runtime proof

- Rank 0 owns experts 0–63; rank 1 owns 64–127 in all 48 layers.
- Non-owned expert modules are removed, so a rank cannot silently execute a
  remote expert locally.
- Rank 0 runs the unchanged router/TEAM masks and sorts token-expert branches
  by owner.
- NCCL `all_to_all_single` performs remote dispatch; only resident experts run;
  a reverse A2A returns weighted results for exact combine.
- A combined-output broadcast keeps the replicated non-MoE path aligned.
- Tests confirm nonzero remote assignments and per-call dispatch/combine work
  conservation.

Against the original full-expert layer on an identical FP16 hidden tensor, the
Python-reference EP2 output had cosine 0.99999994 and relative L2 0.0505%; the
fused path had cosine 1.0 and relative L2 0.0553%. Router logits were identical.

This is a true EP2 reference executor, but not a production DeepEP backend. It
exists to answer whether TEAM semantics and benefits survive physical sparse
ownership.

### Positive control

| Restart | Baseline | TEAM | Speedup |
|---:|---:|---:|---:|
| 2 | 14.546 s | 11.284 s | 1.289x |
| 3 | 14.686 s | 11.589 s | 1.267x |
| 4, warm paired | 12.022 s | 7.751 s | 1.551x |
| **Median** | — | — | **1.289x** |

TEAM's algorithmic benefit therefore survives real remote EP execution. In the
matched structural trace, MoE call count fell 33.3%, but speculative execution
increased remote assignments and hidden payload by 33.6%. TEAM optimizes the
request, not every isolated communication metric.

## Stage C — what dominates after TEAM

### Trivial-fix attack

The reference path initially spent 4.884 s in Python-loop experts. Replacing
only that local execution with the existing vLLM fused expert primitive reduced
clean TEAM time from 7.751 s to 3.338 s, a **56.94% measured reduction**. The
corresponding fused baseline/TEAM pair was 3.936/3.338 s, or 1.179x.

This is the most important falsification: the apparent dominant expert
bottleneck was not a new dLLM phenomenon. It was a deliberately simple
reference backend artifact recoverable by existing engineering.

### Fused residual mass

| Stage | Time | Clean request share |
|---|---:|---:|
| Attention | 1,016.1 ms | **30.44%** |
| Other decoder | 800.0 ms | 23.97% |
| Expert | 342.8 ms | 10.27% |
| Router | 323.5 ms | 9.69% |
| Route preparation | 314.8 ms | 9.43% |
| Combine | 216.6 ms | 6.49% |
| Outside decoder | 153.3 ms | 4.59% |
| Dispatch | 113.8 ms | 3.41% |
| MoE wrapper residual | 51.5 ms | 1.54% |
| LM head | 5.2 ms | 0.16% |

Whole MoE remains 40.84%, but it is distributed across several dependencies.
Attention is the largest individual stage and is not an EP residual. Completely
free dispatch+combine is only a 9.90% physically impossible request bound;
completely free router is 9.69% but would not preserve current decisions.

### Speculative branch structure

TEAM's fused decoder-layer means were 3.869 ms at M=32 and 4.267 ms at M=128.
The fourfold assignment width therefore adds just 0.397 ms per layer. Even a
perfect future oracle that collapses every M=128 speculative layer to M=32 cost
saves only **209.7 ms / 3,337.7 ms = 6.28%**.

Furthermore, 51.00% of speculative assignments repeat the same logical
position/expert route, but only 0.0295% have exactly identical expert inputs.
Within 1% and 5% relative input distance, the fractions are only 1.134% and
3.672%. Exact expert reuse therefore has negligible mass.

## Stage D — candidate oracles

| Candidate | Type | E2E effect / upper bound | Outcome |
|---|---|---:|---|
| Existing fused expert packing | Measured | 56.94% vs Python TEAM | Trivial existing fix |
| Perfect early speculative commitment | Perfect future | **6.28%** | Weak; verification makes smaller |
| Same-route metadata delta reuse | Generous perfect | **4.81%** | Kill below 5% |
| Free router/gating | Unrealistic upper bound | **9.69%** | Not semantics-preserving |
| Free dispatch+combine | Unrealistic upper bound | **9.90%** | Required remote transport; generic prior art |

No new candidate reaches a credible 10% oracle. The only independent value over
5% assumes the winner is known before speculative branch computation and has
no verification/control cost.

## Prior-art attack

Epoch already treats the diffusion block as a compilation unit and carries only
live/new/refresh-required expert work through EP while recomputing live gate
logits.^3 That directly crowds plan reuse and dead-work elimination. DES changes
expert activation through a sequence-level coreset,^4 and REFLEX allocates
expert budgets by refinement state.^5 TIDE exploits temporal expert stability
for I/O-aware offloaded placement,^6 while DICE addresses diffusion-MoE
communication overlap and activation staleness.^7

Consequently:

- reusing route plans is too close to Epoch and below 5%;
- reducing expert sets/budgets overlaps TEAM/DES/REFLEX;
- temporal placement overlaps TIDE;
- stale/cross-step overlap overlaps DICE;
- early speculative commitment is adjacent to TEAM and jump/share speculative
  decoding;
- expert packing is already implemented by fused MoE runtimes.

There is no clean, material novelty gap in the measured residual.

## Final answers

1. **TEAM official trend reproduced?** Yes—1.832x median bounded clean speedup,
   with NFE and active-expert reduction in the paper's direction.
2. **TEAM benefit maintained under true EP2?** Yes—1.289x restart median, 1.551x
   warm paired; 1.179x after an existing fused-expert control.
3. **Dominant post-TEAM bottleneck?** Attention is the largest individual stage
   at 30.44%; aggregate MoE is 40.84% but fragmented. The initial expert
   dominance was a Python-loop artifact.
4. **New >=5%, preferably >=10%, E2E oracle?** One weak 6.28% perfect oracle;
   no credible >=10% independent candidate.
5. **EP4 validation worth doing?** No. The evidence does not support spending
   an EP4 campaign on a candidate whose feasible EP2 headroom is below the
   serious threshold and whose mechanisms collide with adjacent work.

## Evidence boundary and limitations

- Stage A is a bounded subset, not full GSM8K/MATH/HumanEval/MBPP evaluation.
- Stage B uses a reference NCCL sparse A2A executor, not DeepEP.
- EP2 layer correctness meets cosine >=0.9999 and relative L2 <=0.1%; the fused
  control remains a systems falsification, not a full benchmark-quality result.
- Stage shares are CUDA-event attribution from separate runs and are not added
  as if independently removable.
- Candidate values are perfect upper bounds unless marked measured.

## Artifact map

- Detailed reports: `poc_team_positive_ep2/reports/`
- Reproduction scripts: `poc_team_positive_ep2/scripts/`
- Tests: `poc_team_positive_ep2/tests/`
- Compact tables: `poc_team_positive_ep2/results/team_positive_20260911_211456/analysis/`
- Required figures: `poc_team_positive_ep2/results/team_positive_20260911_211456/plots/`

## Sources

1. PKU-SEC-Lab, “[TEAM-MoE-dLLM official repository](https://github.com/PKU-SEC-Lab/TEAM-MoE-dLLM),” accessed September 11, 2026.
2. Wei et al., “[TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](https://arxiv.org/abs/2602.08404),” 2026.
3. Zhu et al., “[Epoch: Compiling Diffusion Blocks for Sparse MoE Serving](https://arxiv.org/abs/2609.09748),” 2026.
4. Chen et al., “[Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs](https://arxiv.org/abs/2602.00879),” 2026.
5. Xia et al., “[REFLEX: Rethinking MoE Inference as Refinement-Aware Compute Allocation in Diffusion Language Models](https://arxiv.org/abs/2608.01784),” 2026.
6. Chen et al., “[TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload](https://arxiv.org/abs/2605.20179),” 2026.
7. Luo et al., “[Staleness-Centric Optimizations for Efficient Diffusion MoE Inference](https://arxiv.org/abs/2411.16786),” 2024.

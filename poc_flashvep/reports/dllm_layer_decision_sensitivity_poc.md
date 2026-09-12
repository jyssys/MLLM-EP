# LLaDA2.0-Flash layer necessity and decision-cost PoC

## Final status

`CHARACTERIZATION-SIGNAL`

The PoC found a strong validation warning—instantaneous decision stability is
not final-trajectory stability—but did **not** find a quality-safe, independent
layer/MoE work-removal opportunity above the 8% prototype gate.  No
optimization prototype or actual speedup is claimed.

## Scope and substrate

- Model: local `inclusionAI/LLaDA2.0-flash` revision
  `LLaDA2.0-flash-744c3f8`, BF16, 32 layers, hidden size 4096, 256 routed
  experts/top-8 plus one shared expert.
- Runtime: dInfer/SGLang, dense TP4 plus routed EP4, DP1, 64 routed experts per
  rank, DeepEP dispatch, owner-rank fused expert execution, reverse combine.
- Hardware: physical H100 GPUs 0,1,2,3 only.
- Best-static configuration: submitted batch 32, `mini_batch_size=32`,
  generation budget 32, block length 32, threshold 0.9, config 42.
- Tasks: the same bounded 32-request GSM8K and HumanEval states for baseline
  and every intervention.

The clean three-restart medians reused from the identical validated substrate
are 6.075 s / NFE 66 / 438.0 token/s on GSM8K and 7.343 s / NFE 86 / 450.6
token/s on HumanEval.  Fresh baselines reproduced all 32 prior outputs and NFE
on both tasks.  Bounded quality was 5/32 GSM8K exact answers and 6/32
HumanEval pass@1; this low-score, 32-token-generation setup is a controlled
causal anchor rather than a full model-quality claim.

Low-overhead component attribution places routed router+dispatch+expert+
combine at 54.37%/56.76% of clean request time and whole measured MoE at
59.52%/62.81% for GSM8K/HumanEval.  The tensor-norm trace had 540--646%
observer overhead, so it is used only for representation analysis.  Causal
intervention runs also execute the original work before replacing its output;
their elapsed time is never treated as speed evidence.

## Experiment design

The evidence chain was:

1. reproduce the exact best-static EP4 baseline;
2. capture layer × refinement stability separately from clean timing;
3. intervene on one full layer or routed-MoE contribution at a time;
4. probe previous-iteration routed output and shared/routed decomposition;
5. attack apparent single-layer safety using contiguous eight-layer groups and
   noncontiguous greedy sets;
6. compare immediate logits/unmask decisions, future NFE/trajectory, final
   token sequence, and task score;
7. map semantically removable work to clean request-level cost;
8. discount work already addressed by an Epoch-like live-row compaction.

Every intervention reaches the same pre-state as baseline before the first
affected wave.  Full-layer bypass is a diagnostic ceiling; it does not model
the K/V-state work a real skipped layer may still require.

## Layer × phase stability

Decision-live rows collapse strongly with refinement, but transformer updates
do not become identity-like.

| task / phase | decision-live ratio | median rel-L2 update | median input/output cosine | median MoE/total update ratio |
|---|---:|---:|---:|---:|
| GSM8K early | 0.904 | 0.3519 | 0.9694 | 0.8370 |
| GSM8K middle | 0.445 | 0.3486 | 0.9672 | 0.8635 |
| GSM8K late | 0.181 | 0.3494 | 0.9639 | 0.8520 |
| HumanEval early | 0.783 | 0.3765 | 0.9628 | 0.8704 |
| HumanEval middle | 0.459 | 0.3587 | 0.9651 | 0.8661 |
| HumanEval late | 0.172 | 0.3570 | 0.9624 | 0.8702 |

Zero of 192 `(task, layer, phase)` cells has median relative-L2 below 0.05.
Absolute update norms decrease with the shrinking ready set, but relative
updates do not.  The hypothesis `late -> nearly identity layers -> safe
refresh skipping` is therefore falsified.

Similarity is also not a useful necessity proxy.  Spearman correlation between
routed-MoE bypass sensitivity and relative-L2 update is 0.002 on GSM8K and
-0.157 on HumanEval; correlation with MoE update norm is 0.255 and 0.034.

## Causal decision sensitivity

### Routed-MoE-only bypass

| task / phase | median live top-1 flip | median accepted-set change | median final exact requests | median NFE |
|---|---:|---:|---:|---:|
| GSM8K early | 13.87% | 2.05% | 25.0/32 | 64.5 |
| GSM8K middle | 19.37% | 2.84% | 10.0/32 | 62.0 |
| GSM8K late | 14.60% | 2.38% | 18.5/32 | 64.0 |
| HumanEval early | 13.78% | 2.74% | 24.0/32 | 90.0 |
| HumanEval middle | 23.40% | 5.85% | 11.5/32 | 86.0 |
| HumanEval late | 14.29% | 3.90% | 19.0/32 | 86.0 |

Middle refinement is the most sensitive phase.  Sampled-logit KL follows the
same direction: GSM8K early/middle/late is 0.0093/0.0567/0.0229 and HumanEval
is 0.0338/0.0929/0.0551.  Late refinement does not contain a growing set of
causally inert routed layers.

### Full layer and layer groups

Full-layer identity bypass has median GSM8K live top-1 flips of
19.3/26.3/20.3% in early/middle/late and leaves only 25/7/14 median final-exact
requests.  No individual full-layer intervention preserves all 32 final
trajectories.  Eight-layer quarter interventions are more destructive, often
leaving zero final-exact requests in the middle phase.  Noncontiguous greedy
sets likewise show that individually low sensitivity does not compose into a
safe multi-layer schedule.

### Shared versus routed experts

Removing routed experts is slightly more sensitive than removing the shared
expert in 16/27 matched representative cells, but both components matter.  On
GSM8K the routed/shared causal scores are 0.0876/0.0707 early,
0.1968/0.1771 middle, and 0.1121/0.1046 late.  The entire shared path accounts
for only 2.98--3.50% of clean E2E, so even its impossible perfect removal fails
the headroom gate.

### Previous-output staleness and future divergence

The most consequential finding is that a stale routed-MoE output frequently
causes zero top-1 or accepted-set changes in the first affected middle-phase
wave but later changes NFE and final output.  The best middle-phase greedy stale
set keeps only 13/32 GSM8K and 9/32 HumanEval final trajectories exact.
Therefore:

> current-step decision unchanged does not imply final diffusion trajectory
> unchanged.

This invalidates a cheap controller based only on immediate acceptance or
top-1 agreement.

## Decision contribution versus physical cost

There is measurable cost/sensitivity heterogeneity, but no stable high-cost,
low-sensitivity region with sufficient safe mass:

- middle-phase routed MoE is expensive and *more* decision-sensitive;
- early stale reuse is safer on HumanEval but accounts for less than 0.5% E2E;
- late routed bypass is somewhat less sensitive than middle, yet tested
  multi-layer savings remain below 3% and change many final trajectories;
- hidden update/cosine does not rank safe work, so a low-overhead deployable
  selector is absent.

Thus the requested mismatch—large physical cost with little causal decision
contribution—does not survive cross-task trajectory validation.

## Phase × layer oracles

Four evidence levels were kept separate:

- O1 sums independently safe single cells and is interaction-unsafe;
- O2 actually executes greedy noncontiguous multi-layer sets;
- O3 actually executes contiguous layer/MoE groups;
- O4 requires final token sequences, NFE, and task outcomes to match baseline.

| oracle/result | GSM8K | HumanEval | two-task mean | evidence boundary |
|---|---:|---:|---:|---|
| best tested benchmark-vector-safe stale set, current-runtime ceiling | 6.89% | 10.65% | **8.77%** | score vector only; trajectory unsafe |
| same set, live-weighted post-Epoch residual | 3.06% | 4.89% | **3.98%** | analytical sensitivity, not measured Epoch |
| best tested fresh routed set | 3.36% | 1.98% | **2.67%** | 16/32 and 19/32 final exact |
| strict cross-task O4 | 0.00% | 0.00% | **0.00%** | no tested nonzero-cost joint policy exact |

The raw 8.77% value is not a quality-safe oracle: the bounded task pass/fail
vector happens to remain unchanged while most token trajectories differ.
Furthermore, it overlaps dead-row work that Epoch-like execution would remove;
the independent residual is below 5% before cache, verification, refresh,
layout, and fallback overhead.

## Prior-art and novelty attack

[DICE](https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html)
already combines stale activations with layer-level selective synchronization
and token-conditional communication for diffusion MoE.  Our language unmask
decision and routed-only intervention are different variables, but the only
raw gate-sized candidate is mechanistically adjacent and empirically unsafe.
[Epoch](https://arxiv.org/abs/2609.09748) uses block-level execution and a fresh
lane to remove non-live work, making the 3.98% live-weighted residual the more
relevant independent ceiling.  [ES-dLLM](https://arxiv.org/abs/2603.10088),
[dLLM-Cache](https://arxiv.org/abs/2506.06295), and
[Window-Diffusion](https://arxiv.org/abs/2601.20332) crowd generic
confidence/update-based skipping and activation reuse.  Finally,
[Layer Collapse](https://arxiv.org/abs/2605.06366) independently warns that
representational redundancy need not imply causal dispensability, consistent
with our direct intervention results.

The theoretically distinct space would couple language-dLLM unmask-decision
sensitivity, routed-MoE-only EP cost, and exact future-safe refresh.  This PoC
does not establish enough headroom or a deployable certificate for that space.

## Answers to the twelve required questions

1. **How do layer updates change from early to late?** Decision-live ratios
   collapse, but relative-L2 updates remain about 0.35--0.38 and do not form a
   late identity region; middle/late transformations remain substantial.
2. **Does layer similarity match decision necessity?** No.  Correlations are
   near zero or weak, and visually stable cells can still alter future output.
3. **What layer/MoE compute changes unmask decisions?** Routed MoE across most
   layers does; middle-phase removal produces the largest top-1, accepted-set,
   KL, and final-trajectory effects.
4. **Do insensitive layers increase late?** No reproducible monotonic increase
   appears.  Late is less sensitive than middle on some metrics, but not safe
   and not enough for a multi-layer policy.
5. **How much E2E is in an insensitive region?** No strict cross-task
   trajectory-safe nonzero region was found.  The strongest score-only stale
   region is 8.77% raw, 3.98% after live weighting, and trajectory unsafe.
6. **Do routed and shared contributions change by phase?** Both remain
   decision-relevant; sensitivity peaks in the middle.  Routed is modestly
   more important in 16/27 cells, but there is no phase where shared alone
   explains decisions safely, and shared cost is at most 3.50% E2E.
7. **What is the strongest perfect phase-layer oracle?** The strict tested
   final-trajectory-exact cross-task oracle is 0.00%.  The largest weak
   benchmark-vector ceiling is 8.77%, not a safe oracle.
8. **Does the current-step oracle survive final quality?** No.  Many stale
   interventions preserve the immediate decision yet change later NFE and
   final sequence; only 13/32 and 9/32 trajectories remain exact for the best
   raw candidate.
9. **How is this different from DICE?** The probes target language-dLLM unmask
   decisions and routed-MoE-only EP work, while DICE targets image diffusion
   with staleness-centric synchronization.  The candidate mechanism remains
   too close and its measured economics/safety do not justify a successor.
10. **Does independent headroom remain after Epoch?** The strongest analytical
    live-weighted residual is 3.98% mean, below the 5% kill gate and before new
    overheads.
11. **Is there a new decision-contribution/latency mismatch?** There is a
    validation mismatch—instantaneous decisions understate future
    sensitivity—but no large safe low-value/high-cost compute region.
12. **Is there evidence for a paper-level training-free method?** No.  The
    characterization is worth retaining, but all candidates fail headroom,
    trajectory safety, or prior-art independence.

## Final decision and recommendation

**Final decision: `CHARACTERIZATION-SIGNAL`.**

Do not build phase-layer skipping, routed stale refresh, or token/expert
selection from these results.  Revisit only if a new exact or conservative
future-trajectory certificate can expose at least 8% request-level *live-work*
headroom on both tasks.  The full evidence, scripts, figures, causal policy
plans, and source patch are under `poc_dllm_layer_sensitivity/`.

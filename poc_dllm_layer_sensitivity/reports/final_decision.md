# Final decision

## Verdict

`CHARACTERIZATION-SIGNAL`

The study found a reproducible and useful systems fact, but not a paper-level
optimization candidate that passes the requested gate.  LLaDA2.0-Flash does
not develop a late-refinement layer identity region, hidden-state similarity is
a poor proxy for causal necessity, and interventions that appear harmless at
the current unmask decision frequently alter the later diffusion trajectory.

No live speedup is claimed.  Causal runs deliberately execute the original
work before replacing its result.  All savings below are cost-mapped upper
bounds against the clean best-static EP4 baseline.

## Decisive evidence

- Decision-live rows fall from 0.904/0.783 in early GSM8K/HumanEval refinement
  to 0.181/0.172 late, but median layer relative-L2 update remains
  0.349/0.357 late.  Zero of 192 task/layer/phase cells has median relative-L2
  below 0.05.
- Routed-MoE bypass is most decision-sensitive in the middle phase, not least
  sensitive late: median live top-1 flips are 19.37% on GSM8K and 23.40% on
  HumanEval in the middle phase.
- Update magnitude does not identify safe layers.  Spearman correlation
  between relative-L2 update and routed-bypass sensitivity is 0.002 on GSM8K
  and -0.157 on HumanEval.
- Previous-iteration routed output can preserve the first affected decision
  while still changing the final trajectory.  This falsifies the shortcut
  `current-step agreement => reusable work`.
- The strongest cross-task, benchmark-vector-preserving tested policy is
  middle-phase stale routed MoE on eight noncontiguous layers.  Its optimistic
  current-runtime ceiling is 8.77% E2E, but it preserves only 13/32 GSM8K and
  9/32 HumanEval final trajectories.  After live-work weighting, the
  independent post-Epoch ceiling is 3.98%.
- No tested cross-task group or greedy policy with nonzero removable cost is
  final-trajectory exact.  The strict tested O4 oracle is therefore 0.00%.

## Gate accounting

| candidate | optimistic current-runtime ceiling | credible independent ceiling | quality result | gate |
|---|---:|---:|---|---|
| full-layer phase refresh | 0% strict cross-task | 0% | no individual full layer preserves all final trajectories | kill |
| fresh routed-MoE selective refresh | 2.67% best tested cross-task multi-layer mean | <5% | only 16/32 and 19/32 final exact | kill |
| middle stale routed-MoE refresh | 8.77% | 3.98% live-weighted | only 13/32 and 9/32 final exact | kill |
| shared-expert removal | 3.50% whole-request component ceiling | <3.50% | both shared and routed paths affect decisions | kill |
| token/expert decision-critical refresh | <5% credible residual | <5% | not escalated after layer gate | kill |

The 8% method-prototype gate is not met by any candidate that is simultaneously
cross-task, trajectory-safe, and independent of already-addressed dead-row
work.  A prototype was therefore forbidden by the working contract and was not
implemented.

## What is novel as characterization, but not yet a method

The strongest observation is a decision-validation mismatch:

> A routed-MoE result can be stale enough to leave the immediate unmask choice
> unchanged, yet perturb the hidden trajectory enough to change later choices,
> NFE, and final output.

This is useful negative evidence for future dLLM controllers: instantaneous
acceptance agreement is not an adequate reuse certificate.  It does not,
however, create a large safe optimization space in this substrate.

## Prior-art boundary

[DICE](https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html)
already studies layer-selective synchronization and token-conditional
communication using stale activations in diffusion MoE.  The language-dLLM
unmask-decision evaluation is different, but our only raw gate-sized candidate
uses the same broad staleness mechanism and fails final-trajectory safety.
[Epoch](https://arxiv.org/abs/2609.09748) removes non-live diffusion work using
block-level execution and a fresh lane; weighting by measured decision-live
ratios leaves only 3.98% mean residual for our strongest stale candidate.
[ES-dLLM](https://arxiv.org/abs/2603.10088),
[dLLM-Cache](https://arxiv.org/abs/2506.06295), and
[Window-Diffusion](https://arxiv.org/abs/2601.20332) further crowd generic
confidence-based skipping, cached refresh, and token-window reuse.  The
remaining language-decision × routed-MoE × EP coupling is conceptually
specific, but it has insufficient measured headroom here.

## Final recommendation

Do not implement a phase-layer refresh controller, routed-MoE stale cache, or
fine-grained token/expert selector from this evidence.  A future revisit would
need a new exact verification signal that predicts *future trajectory safety*,
not merely current-step agreement, and must demonstrate at least 8% residual
request-level headroom after live-row compaction.

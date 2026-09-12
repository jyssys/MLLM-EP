# Adversarial prior-art audit

This audit asks which existing result would invalidate each candidate as an
independent research direction. It intentionally separates a potentially useful
mechanism from a novel paper claim.

## dLLM work elimination and exact execution

Epoch is the closest distributed-system boundary. It treats a diffusion block
as the compilation unit, retains dense logical state, and carries only live,
newly decoded, or refresh-required token–expert work through EP. It reports up
to 2.7× end-to-end improvement on eight H100s.[^epoch] Therefore dead-position
elision, sparse fresh lanes, and repeated route-layout reuse are not available
as independent novelty here. A surviving candidate must reduce the *frequency
or scope of still-required exact global refinement* beyond Epoch.

TEAM changes the decoding algorithm through decoded-token caching, hot/cold
classification, speculative exploration, and limited expert activation.[^team]
The current project starts from vanilla SDAR and treats TEAM only as a later
composition test.

DICE targets diffusion-MoE communication staleness and overlap for image
generation.[^dice] Plain stale communication, selective synchronization, or
moving an existing wait is therefore a direct collision risk for Candidates G
and H.

## Rank-local draft and global verification

The broad idea “use fewer experts as a drafter and verify with the full MoE” is
already occupied. Self-Speculative MoE uses a subset of routed experts for
drafting and a confidence-based target check.[^ssmoe] DraftExpert similarly uses
a reduced-footprint MoE drafter plus exact target verification, although it
trains a lightweight draft expert and targets expert-offloaded edge
execution.[^draftexpert] SpecMoE is training-free self-assisted speculative MoE
inference, primarily motivated by memory/interconnect limits.[^specmoe]

The dLLM verification side is also crowded. SimSD explains why ordinary
token-level verification is invalid under bidirectional masked contexts and
introduces a masking construction that restores valid verification.[^simsd]
Trajectory-level speculative decoding constructs and verifies multi-token
denoising trajectories and reports 30–40% fewer denoising iterations.[^tlsd]

Consequences for RLR-GV:

- “fewer experts as draft” is not novel.
- “multi-step dLLM draft then exact verify” is not novel by itself.
- The only plausible gap is using the *existing physical EP rank partitions as
  four parallel refinement views*, with rank-consensus as a calibrated signal
  for when one exact global verifier can coalesce several refinements.
- That gap matters only if measured rank-view consensus has high precision and
  coverage and the complete draft+verification cost produces material E2E
  headroom. Otherwise Candidates A/B are killed even if the framing sounds new.

## Expert replication and topology selection

Dynamic/predictive expert replication is well established. Recent work predicts
overloaded experts and replicates them for future batches.[^replication]
TIDE exploits temporal expert-activation stability to refresh expert placement
at intervals, although its substrate is GPU–CPU offload rather than fully
resident NVLink EP.[^tide] Consequently BHERC has high novelty risk: block-local
reuse is dLLM-specific, but exact hot-weight replication is not. It survives
only if diffusion-block lifetime yields a qualitatively different copy-amortized
regime and a substantial request-level gain.

Elastic EP degree is adjacent to general hybrid/dynamic parallelism. HDA-MoE,
for example, jointly maps and schedules hybrid parallel execution on
near-memory hardware.[^hda] Candidate D can provide a useful characterization
but needs a dLLM-specific non-monotonic topology crossover and non-trivial state
transition contract to support a paper claim.

## Cost-aware speculative action selection

EcoSpec explicitly scores speculative choices using marginal expert activation
cost.[^ecospec] Thus a generic utility-minus-EP-cost controller or fanout-aware
draft choice is not novel. Any Candidate E/I successor would have to show a
dLLM-specific decision unit—masked-position refinement/global verification
scope—with physical rank cost that cannot be reduced to EcoSpec's AR draft-tree
expert-footprint objective.

## Preliminary collision decisions

| Candidate | Closest collision | Prior-art risk before measurement |
|---|---|---|
| A/B rank-local drafting | Self-Spec MoE + SimSD/trajectory speculation | High |
| C hot replica cache | predictive replication + TIDE | Very high |
| D elastic EP degree | hybrid/dynamic parallel runtimes | High |
| E confidence-gated scope | TEAM/REFLEX + EcoSpec | Very high |
| F temporal delta dispatch | DICE + communication compression | High |
| G layer-gated global EP | Epoch/DICE | Very high |
| H overlap | DICE/generic overlap | Very high |
| I near-tie rank coherence | communication-aware routing + EcoSpec | Very high |

No candidate is promoted on novelty yet. Measurement must first establish a
large independent oracle; an exact mechanism-level search will follow only for
the largest survivor.

## References

[^epoch]: Jianian Zhu et al., “Epoch: Compiling Diffusion Blocks for Sparse MoE Serving,” 2026. https://arxiv.org/abs/2609.09748
[^team]: PKU-SEC-Lab et al., “TEAM: Training-Free Expert-Aware Acceleration for MoE Diffusion Language Models,” 2026. https://arxiv.org/abs/2602.08404
[^dice]: Luo et al., “DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference,” ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html
[^ssmoe]: “Self-Speculative Mixture-of-Experts Decoding,” WWW 2026. https://doi.org/10.1145/3774904.3792218
[^draftexpert]: Dengke Han, “DraftExpert: Expansion-Aware Self-Speculative Decoding for End-Device MoE Inference,” 2026. https://arxiv.org/abs/2607.24434
[^specmoe]: Jehyeon Bang et al., “SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding,” 2026. https://arxiv.org/abs/2604.10152
[^simsd]: Junxia Cui et al., “SimSD: Simple Speculative Decoding in Diffusion Language Models,” 2026. https://arxiv.org/abs/2606.02544
[^tlsd]: Tianxiang Pan et al., “Trajectory-Level Speculative Decoding for Diffusion Language Models,” 2026. https://arxiv.org/abs/2608.27514
[^replication]: Ankit Jyothish et al., “Fast MoE Inference via Predictive Prefetching and Expert Replication,” 2026. https://arxiv.org/abs/2605.11537
[^tide]: Zhiben Chen et al., “TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload,” 2026. https://arxiv.org/abs/2605.20179
[^hda]: Haochen Huang et al., “HDA-MoE: Hybrid Parallelism and Dynamic, Adaptive Scheduling for Mixture-of-Experts with 3D Near-Memory Processing,” 2026. https://arxiv.org/abs/2609.08682
[^ecospec]: Jincheng Xie et al., “Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts,” 2026. https://arxiv.org/abs/2607.12696

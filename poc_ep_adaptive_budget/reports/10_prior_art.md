# Prior-art boundary (source-level / paper-level audit)

This is an adversarial novelty audit, not a measured comparison with other
methods. The current primary experiment uses the LLaDA2.0-Flash stock decoder;
none of the methods below is claimed as a measured baseline.

| Work | What it already covers | Boundary for this PoC |
|---|---|---|
| [Learning Unmasking Policies](https://arxiv.org/abs/2512.09106) | Learns token unmask decisions from confidence with RL; adaptation of position/count is generic dLLM decoding. | A learned budget or `confidence + cost` reward alone is not novel. Must show physical MoE/EP cost shifts the Pareto frontier beyond confidence/live-M. |
| [Beyond Confidence / CCD](https://arxiv.org/abs/2512.02044) | Explicitly adjusts the number of tokens unmasked per step using trajectory consistency. | Dynamic `k_t` is directly prior art. Our only possible distinction is calibrated distributed routed-MoE/EP cost, if it yields incremental value. |
| [LESS Is More](https://arxiv.org/abs/2606.16908) | Training-free commitment rule using confidence and temporal token stability; reduces denoising steps. | Training-free adaptive acceptance by itself is prior art. An EP feature that fails the 3--5% incremental gate cannot support a systems-method claim. |
| [TEAM](https://arxiv.org/abs/2602.08404), [official code](https://github.com/PKU-SEC-Lab/TEAM-MoE-dLLM) | Expert-activation policies and speculative exploration can enable more accepted tokens with fewer activated experts on SDAR. | This PoC leaves top-k, expert outputs and routing untouched. TEAM speedup is not our baseline or claim; overlap remains at aggressive acceptance / NFE reduction. |
| [Epoch](https://arxiv.org/abs/2609.09748) | Liveness/FreshLane compacts already-decoded/stable routed expert work in EP. | Current dense runtime does **not** physically remove finalized rows within a forward. Credits for future fresh-row removal are post-compaction sensitivity, not measured vanilla speedup. |
| [ODB-dLLM](https://arxiv.org/abs/2511.21759) | Adaptive response length and jump-share speculative decoding reduce dLLM work/iterations. | Generic NFE or generation-length reduction is not EP-specific. |

Novelty kill test: if a phase/confidence/live-M budget controller matches the
best EP-cost-aware controller at quality-matched BCT, classify any gain as
`GENERIC-DLLM-SIGNAL`. In particular, this project will not describe simple
dynamic thresholding as a new EP method.

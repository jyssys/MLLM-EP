# Final decision

## Label

**`NO-GEMM-HEADROOM`**

The phenomenon half of the hypothesis passes: refinement drives median M_e from
13--14 toward 2 and late tiny-expert share to 68--79% after live-row compaction.
The method half fails: the strongest whole kernel wins every real replay, an
actual exact subgroup hybrid wins 0/102 case-restarts, and the generous ideal
persistent-scheduler ceiling is only 2.85--3.31% request E2E (1.13--2.12%
post-compaction). No CUDA kernel or E2E integration was implemented.

## Gate table

| gate | result | disposition |
|---|---:|---|
| routed expert request pie | 29.47% / 29.55% | pass |
| refinement changes M_e | strong | pass as characterization |
| different winning strategies | only homogeneous M_e=1 | weak |
| measured hybrid over strongest whole | 0%; 40--47% slower | fail |
| ideal additional request E2E | 3.311% / 2.852% | fail <5% |
| credible additional request E2E | 1.655% / 1.426% | fail <5% |
| post-Epoch-like credible residual | 1.060% / 0.567% | fail <5% |

## Answers to the 20 required questions

1. **Routed expert E2E:** 29.47% GSM8K and 29.55% HumanEval observer-attributed upper bounds.
2. **Global fresh M:** post-compaction medians fall 925.5→419→35.5 and 802→172→15.5 across early/middle/late.
3. **M_e:** median falls 14.25→9→2 and 13→5→2.
4. **Late tiny experts:** yes, M_e<=4 reaches 68.49%/79.43%; dense runtime reaches about 45.7%/64.1%.
5. **Middle coexistence:** yes, but not uniquely; early CV is larger and early also mixes tiny/hot experts.
6. **Heterogeneous execution mass:** permissive criterion covers modeled 59.48%/58.90%; strict criterion 25.15%/31.49%.
7. **Large-M_e winner:** PyTorch grouped among valid measured kernels.
8. **Tiny-M_e winner:** production fused only at homogeneous M_e=1; grouped wins from M_e=2 upward.
9. **Crossover:** between homogeneous M_e=1 and 2, a very narrow region.
10. **Count-only explanation:** incomplete, but matched heterogeneity adds only about 1--2% median latency under the strongest kernel.
11. **Independent heterogeneity factor:** measurable, small and not uniformly directional.
12. **Whole switching:** 0% beyond grouped because grouped wins all real invocations; replacement versus untuned production projects 6.06--6.51% E2E.
13. **Per-expert hybrid:** measured additional gain 0%; best two-subgroup realization is much slower.
14. **Fake serialized oracle:** no. O1 is executed; O2 is a p99-robust critical-rank lower-envelope ceiling preserving work, not summed single-expert times.
15. **Credible RefineGEMM E2E:** 1.66% GSM8K, 1.43% HumanEval; post-compaction 1.06%/0.57%.
16. **8--12% gate:** no; even ideal O2 is below 5%.
17. **Middle concentration:** absent. Measured hybrid wins nowhere; ideal residual is not a middle-specific large gain.
18. **MonoMoE distinction:** BF16 mixed-strategy persistent scheduling is technically narrower, but the kernel space is crowded and economics fail.
19. **Implementation value:** exact implementation is feasible in principle, unjustified under this gate.
20. **Paper-level headroom:** no for RefineGEMM on this substrate; retain the M_e atlas and existing-backend tuning result only.

## Recommended action

Do not build RefineGEMM. If operational performance matters, integrate or tune
the existing PyTorch/CUTLASS grouped path and measure full-model E2E; classify
that as runtime engineering. Reopen research only with a genuinely stronger
tiny/medium BF16 primitive that raises the one-launch *additional* oracle above
8% on both tasks.


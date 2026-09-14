# Final decision

## Label

`NO-KERNEL-HEADROOM`

The characterization premise is true: an Epoch/FreshLane-like compacted
LLaDA2 block spans fresh M=1--1024, and roughly two thirds of layer-waves are
medium-small/medium. The kernel premise is false on this substrate: existing
DeepEP low-latency wins every real route and every matched control, and the
credible new-path increment over it is only 2.29%/2.10% request E2E. Even a
zero-control impossibility bound is only 4.40%/4.39%.

## Required questions

1. **Fresh-M distribution after compaction?** GSM8K p10/p25/p50/p75/p90/max
   is 4/30/164/406/483/1024; HumanEval is 5/30/102/247/440/802.
2. **Different physical regimes?** Yes. Early medians are 925.5/802 rows and
   late medians are 35.5/15.5; rows/expert falls from 14.25/13 to 2/2.
3. **Existing winner by regime?** Valid low-latency DeepEP beats normal-fresh
   in all 25 real and all 15 controlled cases, from M=1 to 1024.
4. **Uncovered medium/small regime?** No. It is high-mass, but LL covers it.
5. **Request E2E mass there?** All compacted LL communication sums to
   259.1/298.3 ms, 4.37%/4.11% of current clean GSM8K/HumanEval; medium plus
   medium-small alone is about 3.07%/2.75%. This bounds any third-path value.
6. **Useful transfer vs runtime tax?** The measured-payload bound is
   55.5/51.7 ms versus 259.1/298.3 ms LL communication. The difference is an
   optimistic mixture of control, local movement, and required protocol—not
   all safely removable. O3 removes only the profiling-credible part.
7. **Best-existing dynamic oracle?** O0 gives 5.16%/6.08% over a hypothetical
   post-compaction normal path, but it degenerates to static LL.
8. **Credible specialized-kernel oracle?** 2.29% GSM8K and 2.10% HumanEval
   incremental over O0.
9. **Does it exceed 12%?** No; it is more than fivefold below the gate.
10. **What overhead would need removal?** Remaining LL issue/progress,
    synchronization, local packing/movement, and launch floor—not primarily
    normal layout. Removing all control still fails 5%.
11. **Can fixed-contract CUDA reproduce correctness?** Plausible from the
    understood contract, but unproven because the gate prohibited a kernel.
12. **Does RefineEP beat normal on target shapes?** No RefineEP exists.
    Existing LL beats normal by median 24--63% depending on shape.
13. **Does it beat every LL/V2 alternative?** No. It was not built; LL is the
    winner and V2 is outside the local software contract.
14. **Is output exact enough for trajectory equivalence?** Existing LL
    communication has max rel-L2 0.000897 and exact assignment counts. No
    RefineEP trajectory claim is made.
15. **Does microbenchmark gain survive integration?** Not tested because the
    analytical gate failed.
16. **Is speedup from a new path rather than switching?** No new speedup exists.
    O0 is entirely an existing-path result.
17. **Which new kernel mechanisms matter?** None were promoted. Cached handles
    save ~40--46 us but do not beat LL for 23/25 real cases.
18. **Is persistent block-lifetime execution necessary?** No evidence supports
    it; even O1 zeroes control and fails the kill gate.
19. **Does EP8 preserve the phenomenon?** Not tested because EP4 did not pass.
20. **Distinct from generic fused/persistent kernels?** The workload atlas is
    distinctive; the proposed fixed-buffer/persistent/streaming mechanisms are
    not sufficiently distinct from DeepEP/StreamEP/FlashMoE/mKernel.

## Recommendation

Do not implement RefineEP on 4xH100 EP4 for this contract. If liveness
compaction becomes available, use/validate the existing low-latency DeepEP
path first. Reopen kernel research only on a new substrate where the
best-existing path leaves a measured or credible request-level gap >=12%.

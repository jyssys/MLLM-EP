# LLaDA2.0-Flash 100B Expert-Parallel Scaling and Method PoC

## Executive decision

**FINAL LABEL: `CHARACTERIZATION-ONLY`**

This PoC corrects the earlier SDAR substrate mistake: LLaDA2.0-Flash is a real
capacity-driven sparse deployment, and true EP4 fits and runs without CPU
offload. However, after tuning the strongest static baseline, excluding work
already removed by Epoch, and charging DeepEP's measured fixed cost, no new
candidate has a credible request-level E2E oracle of 8% or more. The strongest
novelty-eligible upper bound is 5.34%.

The practical finding is still valuable: EP economics are governed at least as
much by execution granularity and shared runtime state as by nominal batch.
The existing dInfer microbatch knob changes EP4 BCT by 47.36%, dwarfing all
new-method residuals.

## Scope and immutable setup

- Hardware: physical NVIDIA H100 80 GB GPUs 0,1,2,3 only, fully NVLink connected.
- Model: `inclusionAI/LLaDA2.0-flash`, revision
  `744c3f8c6c8317d2377d6d16d8a3d4be2caef563`.
- Official dInfer: commit `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`.
- Executable stack: dInfer + SGLang 0.5.3.post1 + torch 2.8.0/cu128 +
  DeepEP 1.2.1 + BF16 Triton experts.
- Quality anchor: temperature 0, block length 32, threshold decoder, matched
  GSM8K/HumanEval prompts.
- No CPU expert offload, expert replication, EPLB, semantic routing change, or
  production optimization was used.

The official [dInfer repository](https://github.com/inclusionAI/dInfer)
documents LLaDA2.0-Flash on four GPUs but constructs tensor parallelism from
that GPU list; four GPUs are not evidence of sparse EP. The
[LLaDA2 project](https://github.com/inclusionAI/LLaDA2.X) describes the flash
model as a 100B MoE and the official SGLang-based engine. The downloaded
checkpoint, not memory or a model card, is the source of truth here.

## Model and runtime truth

The actual checkpoint has 102,889,705,216 parameters and occupies
205,779,410,432 bytes. It has 32 layers, hidden size 4096, 256 routed experts,
top-8 routing, one shared expert, MoE intermediate size 1024, eight routing
groups/top-four groups, 32 query heads, four KV heads, and a 32,768 context
limit. Routed experts account for 97.1% of parameters. A full BF16 copy cannot
fit in one 80 GB H100.

### Official path

`TP4 / EP1 / DP1 / no sequence parallelism`: attention, dense layers, shared
experts, and all 256 routed experts are tensor-sharded.

### Bounded true-EP path

`dense TP4 + routed EP4 / DP1 / no sequence parallelism`: each rank owns 64
complete contiguous routed experts. Each sparse block partitions source rows,
uses [DeepEP](https://github.com/deepseek-ai/DeepEP) normal-mode remote
dispatch, executes owner-local BF16 Triton experts, reverse-combines, applies
the full shared expert to disjoint source rows, and exactly gathers those rows.

Two official loader defects required substrate-only repair: the loader retained
an unfiltered expert shard, and unquantized expert fusion ran only on the rank
owning global expert zero. Independent checkpoint replay caught the latter.
After repair, 75.1% of assignments are remote, all receiver IDs are in local
range 0--63, and layer-1 routed relative L2 is 0.373%.

## Correctness and quality

The EP4 path uses 56.862 GiB/rank versus TP4's 56.999 GiB/rank. Layer 0 is
bitwise identical. Final-layer hidden cosine is 0.999146 (relative L2 4.16%),
reflecting accumulated BF16 reduction-order differences. It is not claimed to
be bitwise trajectory-equivalent.

| Task | TP4 | routed EP4 | repetitions |
|---|---:|---:|---:|
| GSM8K bounded 32 | 5/32 | 5/32 | 5 paired restarts |
| HumanEval bounded 32 | 4/32 | 4/32 | 1 paired screen |

Surface outputs differ more often than task scores (GSM8K exact strings
15/32), so task metric rather than string identity is the gate.

## Topology and batch scaling

For generation-128 GSM8K, TP4 median BCT is 34.396 s and EP4 is 38.396 s.
Paired EP gains over five restarts are +29.51%, -27.42%, -33.80%, +8.02%, and
-54.45% (median -27.42%). The sign instability prevents a topology winner
claim. Unrelated jobs occupied GPUs 4--7 and may have introduced shared
host/NVSwitch state; they were not touched.

With `mini_batch_size=4`, nominal paired EP gains progress from -29.94% at
submitted batch 1 to +20.75% at batch 32, but the large-batch result contains
a -25.91% restart. It is not robust.

The physical microbatch sweep is decisive:

| Topology | best mini batch | median BCT | tokens/s |
|---|---:|---:|---:|
| routed EP4 | 8 | 6.904 s | 238.41 |
| TP4 | 16 | 5.641 s | 291.86 |

EP4 mini4→mini8 reduces BCT 47.36%; mini16 regresses to 8.040 s. TP4 continues
improving through mini16. There is a real decomposition-dependent granularity
crossover, but the existing option recovers it completely.

## Denoising and temporal structure

At EP4 mini8, decision-live/physical rows fall from 82.88% early to 49.30%
middle and 13.31% late; overall only 41.56% of executed rows are live. Yet the
physical MLP share stays near 50--54%. This validates the motivation of
[Epoch](https://arxiv.org/abs/2609.09748), whose Expert Atlas, Liveness, and
FreshLane already compile diffusion blocks and compact live expert work.

Fresh LLaDA2 temporal routing is neither assumed from SDAR nor simply stable:

| lag | exact top-k set | expert overlap | destination overlap | rank-load cosine | critical-rank persistence |
|---:|---:|---:|---:|---:|---:|
| 1 | 29.11% | 74.29% | 93.18% | 0.9942 | 87.18% |
| 2 | 21.96% | 67.41% | 91.48% | 0.9908 | 83.84% |
| 4 | 15.11% | 58.28% | 89.30% | 0.9843 | 79.11% |
| 8 | 7.76% | 46.07% | 86.51% | 0.9713 | 72.21% |

Fine branches change while coarse destination/load shape persists. This is a
strong predictor signal but a weak work-removal signal.

## Production bottleneck boundary

Observer-light block timing assigns 53.06% of sequential CUDA block time to
MLP and 27.36% to attention. A detailed inner-MoE trace splits MLP into router
10.13%, dispatch 26.56%, expert 34.13%, combine 9.80%, shared expert 8.29%, and
exact row gather 11.10%.

These are not clean E2E shares: block-only instrumentation costs +19.46% at the
representative point and detailed tracing costs +328.27%. They are used only
to calibrate optimistic upper bounds. Rank rows are joined, never summed.

The DeepEP primitive sweep finds dispatch+combine median 0.257--0.424 ms from
global M=4 to 512. At M=256 only 20.88% lies above the observed fixed floor.
Payload savings therefore do not translate linearly into latency.

## Candidate tournament

| Candidate | perfect E2E | credible/feasible E2E | disposition |
|---|---:|---:|---|
| A. Expert work packing | 1.80% | 1.80% | KILL |
| B. EP wave coalescing | 47.36% measured | 47.36% | existing knob; incremental only |
| C. Block-scoped replication | 14.33% | ≤2.99% before copy | KILL |
| D. Perfect rank balance | 5.34% | ≤5.34% | KILL |
| E. Phase topology | unidentifiable | 0% | dual layout does not fit |
| F. Fresh-work compaction | 25.00% | excluded | Epoch collision |
| G. Route-plan reuse | 5.38% | 1.57% | KILL |
| H. Attention/EP overlap | 19.29% zero-contention | 0% established | prior-art/no live causality |
| I/J. Locality/support grouping | none | 0% | not bottleneck-activated |
| K. Phase precision | unmeasured | 0% | no quality-valid path |

[MoE Parallel Folding](https://arxiv.org/abs/2504.14960) establishes that
dense and MoE layers can prefer different parallel mappings, but in a
large-scale training setting. Here the concrete TP/EP layouts each already use
about 57 GiB/rank, so retaining both or moving tens of gigabytes per phase is
not a feasible low-cost inference switch.

Communication-cost sensitivity makes C weaker: its 2.99% upper bound becomes
2.24%, 1.50%, and 0.75% at communication multipliers 0.75, 0.5, and 0.25. D
remains 5.34%, still below the 8% HOLD gate and in a crowded load-balancing
space.

## TEAM transfer

`TEAM-PORT-NOT-ESTABLISHED`. TEAM is SDAR-specific and secondary in this spec.
Because no vanilla novelty-eligible candidate reached the HOLD gate, adapting
TEAM would not rescue the primary method question and was not attempted. This
is not a TEAM failure result.

## Answers to the final questions

1. Official four-GPU runtime: TP4, not sparse EP.
2. True EP4 without CPU offload: yes, 64 experts/rank and 56.86 GiB/rank.
3. Feasible comparison: TP4 versus dense-TP4+routed-EP4; TP2×EP2 unsupported.
4. EP crossover: nominal at batch 8+, but not stable; best-static TP4 wins the
   batch-16 screen.
5. Low/high batch: startup/fragmentation → amortization → oversized-wave
   regression.
6. Denoising phase changes logical work greatly, but physical runtime does not
   exploit it.
7. Temporal persistence: coarse load yes, exact route no; residual oracle small.
8. Strongest new oracle: perfect rank balance, 5.34% E2E.
9. Training-free/TEAM-independent: yes in formulation, insufficient in value.
10. Epoch complement: no large independent candidate found.
11. Faster backend durability: communication-derived gains rapidly vanish.
12. EP8 follow-up: not justified for this method direction; only for clean-node
    deployment characterization.

## Final recommendation

Do not invest in a new LLaDA2-specific EP method from these candidates. For
deployment, tune model-forward microbatch per topology and remeasure TP4/EP4
on an isolated node. For research, either build directly on Epoch with an
independent post-FreshLane bottleneck, or move to a substrate/regime where the
best-static residual itself exceeds 8--12%.

## Reproducibility

- Working data and scripts: `poc_llada2_flash_ep/`
- Runtime patch: `poc_llada2_flash_ep/patches/dinfer_llada2_true_ep4_and_trace.diff`
- Raw per-rank traces remain local due to size; derived CSV/JSON summaries are
  versioned.
- Detailed subreports: `poc_llada2_flash_ep/reports/`

# RefineGEMM refinement-adaptive expert execution PoC

## Executive decision

**Final label: `NO-GEMM-HEADROOM`.**

LLaDA2.0-Flash refinement does reshape the per-expert row distribution: under
the exact-route live-row sensitivity, median M_e falls from 13--14 early to 2
late, and active experts with M_e<=4 rise to 68.5% GSM8K / 79.4% HumanEval.
Routed expert execution is also a large 29.5% request-critical pie. Those facts
make the question valid.

The decisive method test is negative. PyTorch `torch._grouped_mm` beats the
installed production vLLM fused expert on every one of 102 real
scope/case/restart comparisons. The only alternative crossover is a homogeneous
M_e=1 case, where production fused is 5.17% faster. Exploiting that crossover
with an actually executed exact two-subgroup hybrid never wins: its median
penalty is 40.4% for dense and 47.4% after live-row compaction. A deliberately
generous one-launch perfect-shape ceiling is only 3.311% GSM8K / 2.852%
HumanEval request E2E; the credible target is 1.655% / 1.426%. No CUDA kernel or
full-model method integration was implemented.

## Substrate and evidence

- Model: local `inclusionAI/LLaDA2.0-flash` revision directory
  `LLaDA2.0-flash-744c3f8`, BF16, hidden 4096, 32 layers, 256 routed
  experts/top-8, intermediate 1024 and one shared expert.
- Runtime: dense TP4 plus routed EP4, 64 experts/rank, DeepEP normal dispatch,
  owner-local fused expert and reverse combine.
- GPUs: new measurements use only physical H100 4--7 under
  `CUDA_VISIBLE_DEVICES=4,5,6,7`; all peers are NV18.
- Strongest request setting: submitted batch32, mini32, gen32/block32,
  threshold0.9. Prior three-restart medians are 6.075 s/NFE66 GSM8K and
  7.343 s/NFE86 HumanEval. A fresh GPU4--7 clean run provides a portability
  anchor with NFE66 and 32/32 exact answer-field agreement; its single-run
  timing is excluded. Operator and request evidence remain explicitly separated.
- Kernel evidence: three independent process restarts, warmup5 and 30
  randomized measured repetitions for dense and compacted corpora.

The live-row compacted corpus is a future-known Epoch-like sensitivity, not a
measured Epoch result. Full environment/evidence boundaries are in
[00_environment_and_baseline.md](../../poc_refinegemm/reports/00_environment_and_baseline.md).

## Execution pie and regime atlas

| task | routed expert share | early fresh M / median M_e / tiny<=4 | middle | late |
|---|---:|---|---|---|
| GSM8K | 29.47% | 925.5 / 14.25 / 27.05% | 419 / 9 / 35.15% | 35.5 / 2 / **68.49%** |
| HumanEval | 29.55% | 802 / 13 / 29.34% | 172 / 5 / 46.90% | 15.5 / 2 / **79.43%** |

The hypothesized early→heterogeneous-middle→tiny-late progression is only
partially correct. Late is clearly tiny-heavy, but M_e CV is highest early and
falls monotonically; both early and middle mix tiny and hot experts. A
permissive tiny+large definition covers about 59% of modeled compacted expert
time, so lack of mass does not explain the negative kernel result.

Details: [operator pie](../../poc_refinegemm/reports/01_operator_breakdown.md),
[row trace](../../poc_refinegemm/reports/02_expert_row_trace.md), and
[atlas](../../poc_refinegemm/reports/03_refinement_regime_atlas.md).

## Strongest existing envelope

At the owner-local operator, PyTorch grouped improves over the current untuned
vLLM fused expert by 22.08% GSM8K / 20.52% HumanEval on dense real replays. An
Amdahl projection gives 6.51% / 6.06% request E2E, before integration and
packing overhead. This is an existing-backend engineering opportunity, not
measured E2E and not a new heterogeneous scheduler.

At equal total assignments and active-expert count, the high/low heterogeneity
latency ratio is only 1.0205 dense and 1.0139 compacted under the strongest
kernel. M_e shape has independent cost, but its controlled median effect is
small. Full data: [kernel envelope](../../poc_refinegemm/reports/04_existing_kernel_envelope.md),
[single/homogeneous sweep](../../poc_refinegemm/reports/05_single_expert_sweep.md),
and [matched replay](../../poc_refinegemm/reports/06_whole_invocation_replay.md).

## Oracle tournament

| comparison | GSM8K E2E | HumanEval E2E | evidence |
|---|---:|---:|---|
| production→existing grouped | 6.505% | 6.063% | operator replay + Amdahl projection |
| O0 switching beyond strongest whole | 0% | 0% | grouped wins every real replay |
| O1 measured two-subgroup hybrid | 0% | 0% | actual gather/two-launch/scatter |
| O2 perfect one-launch shape removal | **3.311%** | **2.852%** | p99-robust critical-rank lower envelope |
| O3 credible 50%-capture | **1.655%** | **1.426%** | unimplemented target |
| post-compaction O2 residual | 2.119% | 1.134% | modeled sensitivity |
| post-compaction O3 residual | 1.060% | 0.567% | modeled sensitivity |

O2 is not a fake sum of serial single-expert calls. It preserves all expert
assignments/useful FLOPs and replaces real critical-rank expert time with a
nearby count-matched lower envelope, thereby optimistically deleting all
positive shape deviation and even some noise. Its failure below 5% is decisive.

Details: [switching](../../poc_refinegemm/reports/07_switching_oracle.md) and
[hybrid oracle](../../poc_refinegemm/reports/08_hybrid_oracle.md).

## Correctness and evidence boundaries

The tested existing paths use identical packed BF16 inputs, weights, SwiGLU,
route weights and output order. Recorded maximum relative L2 and max absolute
difference are zero. No RefineGEMM kernel entered a generation trajectory, so
the report makes no new-method quality or measured E2E claim. See
[correctness](../../poc_refinegemm/reports/10_refinegemm_correctness.md) and
[E2E disposition](../../poc_refinegemm/reports/12_end_to_end_integration.md).

## Prior-art attack

The remaining design space is crowded. PyTorch already exposes jagged-M
[grouped_mm](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html);
CUTLASS documents [persistent grouped GEMM scheduling](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/005_grouped_gemm_contiguous_offset.html);
[SonicMoE](https://github.com/Dao-AILab/sonic-moe) is an H100 BF16 tile/IO-aware
kernel; [MonoMoE](https://arxiv.org/abs/2609.04244) targets weight-major
persistent decode execution; and [DA-MoE](https://arxiv.org/abs/2607.23099)
selects kernels using routing distribution/skew. One mixed-strategy persistent
BF16 launch could be technically narrower, but novelty cannot rescue a sub-5%
perfect E2E ceiling. Full matrix:
[13_prior_art_and_novelty.md](../../poc_refinegemm/reports/13_prior_art_and_novelty.md).

## Final answer

There is a real and reusable characterization result—dLLM refinement changes
M_e sharply, and late live work is tiny-expert-heavy. There is also an immediate
runtime-engineering result—the current E64/N1024 vLLM default is weaker than an
existing grouped path. But there is no paper-level RefineGEMM headroom over the
strongest valid backend on this substrate. The required 20-question disposition
and exact gates are in [final_decision.md](../../poc_refinegemm/reports/final_decision.md).

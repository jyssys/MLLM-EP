# SERE / Libra / MoDES successor mining — CPU-only interim report

> **2026-09-08 resumption in progress,14:30 KST:** physical4,5,6,7 only.
> Official MoDES calibration and SERE quality confirmation are complete; the
> six-condition clean EP screen and port/length/no-op controls have run. Full
> native Libra eight-group exact-prediction controls are running, followed by
> corrected-MoDES clean-serving reruns. See `GPU_WORK_DEFERRED.md` and resumed
> checkpoint/control notes. The CPU-only body below is the **historical release
> checkpoint**, not the current status. No final successor winner/verdict yet.

Status: **CPU_SCREENING_COMPLETE; GPU_VALIDATION_DEFERRED; NOT_FINAL_RESEARCH_VERDICT**.

The user released GPUs on 2026-09-07 at approximately 17:37 KST and subsequently
authorized CPU work only. All owned GPU jobs and the queued handoff were stopped.
No GPU experiment, burn, or automatic reacquisition is scheduled. GPU work must
be requested separately. This checkpoint does not satisfy or waive the original
all-three screening, cross-model, clean-serving, and successor-prototype gates.

## Bottom line

There is no defensible `FOUND_SUCCESSOR_STRONG_GO` or final winner yet. The most
visible exploratory quality loss is in SERE's small-batch ChartQA decode path,
but it is strongly reduced by a simple batch/threshold change and has not been
confirmed with the official-norm calibration table. Libra's real predictor has
lower Vision recall in matched samples, yet the actual supplied planner absorbs
much of that difference. MoDES's pilot OCR trade-off is material but already
acknowledged by the paper; full-calibration held-out performance remains unknown.

These are reasons to sharpen the next tests, not reasons to declare three
method failures. None currently establishes quality-matched request E2E headroom.

The assumption-checking and surgical-change skills influenced this checkpoint:
exploratory evidence is kept separate from confirmation, and minimal ports do not
modify the working vLLM environment or silently replace the author algorithms.

## Provenance and evidence boundaries

Branch: `flashvep/top-tier-successor-mining`.

Result root:
`poc_flashvep/deepep_revalidation/results/top_tier_successor_mining_20260907_135950/`.

Primary checkpoint: `Qwen/Qwen3-VL-30B-A3B-Instruct`, snapshot
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16. Prior live EP sanity used
TP2/DP2/EP4, vLLM 0.20.0 V1, DeepEP HT 1.2.1+73b6ea4, TritonExperts, linear
placement, eager execution, DBO/EPLB/prefix cache off. Source and runtime hooks
verified the DeepEP prepare/finalize path; this small instrumented sanity run is
not a clean performance reproduction.

| Reference | Pinned source | Reproduction boundary at release |
|---|---|---|
| [SERE](https://arxiv.org/abs/2602.07616) | `8512f39b108cd1fa04e9261a5b6846fb173989e4` | Official CUDA rerouter tested; exploratory Qwen-VL quality; clean EP request performance pending |
| [Libra](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9ff1ac9a659085fed0735362cafe5e53-Abstract-Conference.html) | Author supplementary patches on SGLang `023288645b80fb41b3eed55fd413dd69a7904593` | Actual Cython planner and four-layer native four-GPU functional path tested; full VL path pending |
| [MoDES](https://arxiv.org/abs/2511.15690) | `933b3ccac3c23e01c763771f1a7d60c4e7ed13a4` | Official decision semantics and importance calibration tested; full threshold search interrupted |

The user-supplied Libra supplementary files supersede the initial public-code
availability limitation. Their local provenance is recorded in
[SUPPLEMENT_AUDIT](../top_tier_successor_mining/LIBRA/SUPPLEMENT_AUDIT.md).
Native Libra uses its AllGather/local/remote/AllReduce path, **not DeepEP**.
Its author runtime/parallelism must not be presented as the primary vLLM topology.

Quality replicas, local planner diagnostics, native distributed functional tests,
and online EP measurements are distinct evidence classes. No concurrent quality
calibration/native functional timing is used as a speedup result. Kimi did not run.

## SERE

Core insight: exploit expert similarity to substitute work toward an already
selected batch expert set. The batch union and similarity decision are important
parts of the execution contract, not interchangeable with per-token pruning.
See [paper audit](../top_tier_successor_mining/SERE/PAPER_AUDIT.md) and
[code audit](../top_tier_successor_mining/SERE/CODE_AUDIT.md).

128 official rerouter identity tests passed. Exploratory quality used an
FP32-Gram similarity table. The confirmatory 400×128 FineWeb table using the
official rounded BF16 subtraction/norm arithmetic is saved, but **its held-out
quality and clean serving evaluation have not run**.

ChartQA, 128 examples, paired accuracy changes from the corresponding vanilla:

| Exploratory policy | Accuracy | Change | Image-cluster bootstrap 95% CI for change |
|---|---:|---:|---:|
| B1 vanilla | 89.06% | — | — |
| B1 S2/rho .5, all phases | 69.53% | −19.53 pp | [−27.34, −13.28] pp |
| B1 S2/rho .5, decode only | 69.53% | −19.53 pp | [−27.34, −12.50] pp |
| B1 S2/rho .5, prefill only | 88.28% | −0.78 pp | [−3.91, +1.56] pp |
| B16 S2/rho .5 | 88.28% | −1.56 pp | [−3.91, 0.00] pp |

B16's own vanilla is 89.84%. These are HF greedy short-answer transfer probes,
not reproductions of the original paper's stochastic batch-16 evaluation or
single-H20 fixed-input/output speed protocol. In GQA64, S2/rho .5 had no average
accuracy change, with a wide confidence interval. The evidence does not establish
a causal Vision-specific similarity problem.

CPU follow-up compared both calibration tables on 6,336 sampled routing
conditions from 24 real requests and eight layers. S2/rho .5 mapping differences
were about 0.85–2.31% of assignments across the sampled sizes/modalities. These
are subsampled prefill routes, not real decode measurements. The result is enough
to require confirmation, not enough to attribute the task-quality loss to
calibration arithmetic.

The obvious batch/threshold controls substantially weaken a broad successor
claim. Natural-EOS differences also mix formatting, termination, and genuine
content errors; first-line rescoring is a diagnostic, not a replacement benchmark.
**Successor oracle: NOT_MEASURED.**

![Exploratory SERE phase/batch controls](../deepep_revalidation/results/top_tier_successor_mining_20260907_135950/analysis/cpu_release_checkpoint/plots/sere_exploratory_phase_batch.png)

## Libra

Core insight: look ahead with an expert predictor, plan replication and token
sharding, and overlap prefetch/execution. The supplied code's actual planner is
the baseline; an earlier paper-only surrogate was not run as primary evidence.
See [paper audit](../top_tier_successor_mining/LIBRA/PAPER_AUDIT.md) and
[code audit](../top_tier_successor_mining/LIBRA/CODE_AUDIT.md).

The unchanged Cython planner passed 48 invariants. Native four-GPU execution with
real Qwen text weights and four layers matched the first greedy token in 24/24
rank comparisons, but was not bit-exact: minimum logit cosine 0.9997333, maximum
absolute difference 0.40625, maximum relative L2 0.0231472. This is not full-model
parity or MLLM quality validation. The VL bridge passed six CPU boundary checks
for embedding identity, MRoPE arithmetic and DeepStack residual ordering; GPU
capture and full 48-layer execution were cancelled before launch.

CPU matched-route analysis uses the same quartet/layer/per-source token budget:
1,081 conditions from 23 distinct quartets. Median Vision minus Text:

- Predictor recall: **−6.64 pp**.
- Predicted local fraction: **−7.23 pp**.
- Additional local fraction attainable by perfect prediction: **+0.78 pp**.

Across all planner conditions, perfect prediction improves local fraction by
0.78 pp median and 3.13 pp p90. Vision's corresponding p90 is 3.48 pp. Some cases
have larger differences, but those are not request-weighted execution savings.
These conditions share requests/layers and are not 1,081 independent serving runs.

The inference is narrow: recall alone overstates the evidence for a material
planner failure. Actual overlap, wrong-prediction cost, and direct request/TTFT
headroom remain unmeasured. Equal-source-length/static-buffer assumptions are
port-scope questions until a faithful runtime baseline demonstrates a failure.
**Successor oracle: NOT_MEASURED.**

![Libra matched planner diagnostic](../deepep_revalidation/results/top_tier_successor_mining_20260907_135950/analysis/cpu_release_checkpoint/plots/libra_matched_recall_vs_plan.png)

## MoDES

Core insight: calibrate modality/layer importance and expert-skipping thresholds.
Its modality awareness is already the method, not successor novelty. The original
ablation returns normalized MLP input rather than zero; BF16 normalized top-k
weights and no after-drop renormalization are preserved in the decision audit.
See [paper audit](../top_tier_successor_mining/MODES/PAPER_AUDIT.md) and
[code audit](../top_tier_successor_mining/MODES/CODE_AUDIT.md).

The full 1,024-question GQA importance calibration is complete. Calibration and
held-out GQA images are disjoint; 1,024 questions must not be called 1,024 images.
The official-style 100-grid threshold search was stopped at user GPU release:
301 unique measured threshold pairs, first target complete, second target partial.

| Calibration point | Actual skipped assignments | Official answer-position KL | Status |
|---|---:|---:|---|
| 70% target | 71.33% | 0.01108 | Completed target |
| Best observed 85%-feasible point | 85.27% | 0.01702 | Search unfinished; not certified optimum |

Held-out performance for these thresholds is **unknown**. Earlier smaller/coarser
pilot calibration produced ChartQA128 losses of 3.13 pp and 11.72 pp at aggressive
settings. The paper itself reports an aggressive OCR quality trade-off, so this
does not yet identify a new failure after faithful calibration.

CPU monotonicity analysis checked 462 observed fixed-other-threshold comparisons:
10 skip-rate decreases, five at least 0.1 pp, none at least 1 pp; maximum 0.903 pp.
The largest reversals occur in high-KL regions, not the good-quality frontier.
They do not establish material search failure. Missing grid points were not filled
with simulated model results. **Successor oracle: NOT_MEASURED.**

![Partial MoDES calibration](../deepep_revalidation/results/top_tier_successor_mining_20260907_135950/analysis/cpu_release_checkpoint/plots/modes_partial_calibration.png)

## Prior-art attack and ranking

Generic expert substitution overlaps SERE and related expert-sharing work;
prediction/replication improvements must distinguish [PROBE](https://arxiv.org/abs/2602.00509)
and the actual Libra pipeline. OCR-sensitive expert allocation and modality-aware
reduction are already discussed by [AnyExperts](https://arxiv.org/abs/2511.18314),
[MoDES](https://arxiv.org/abs/2511.15690), and [MACS](https://arxiv.org/abs/2605.05225).
These are candidate-killing alternatives, not claims of exact equivalence.
Mechanism comparisons are in the
[prior-art matrix](../top_tier_successor_mining/PRIOR_ART_MATRIX.md).

No final #1/#2/#3 ranking or numerical research score is assigned. Blank scoreboard
fields mean **unmeasured**, not zero. Selecting SERE merely because its exploratory
failure is numerically largest would violate the equal-screening milestone.

## CPU work completed; GPU work deferred

This checkpoint adds paired quality/uncertainty consolidation, matched native
Libra planner analysis, calibration-arithmetic route sensitivity, partial MoDES
frontier and monotonicity analysis, bridge unit tests, plots and reproduction
boundaries for all three papers. No GPU was used for this continuation.

The [GPU work list](../top_tier_successor_mining/GPU_WORK_DEFERRED.md) records each
remaining question, prerequisite and planning duration. The minimum decisive work
is official-table SERE confirmation plus clean natural-EOS EP serving; full native
Libra text/VL parity followed by overlap/TTFT comparisons; and completed MoDES
calibration followed by held-out quality and fast-path serving parity. Kimi and
successor mechanisms are conditional on a material surviving failure.

The saved [GPU policy](../top_tier_successor_mining/GPU_EXECUTION_POLICY.json)
disables GPU runs, burn and automatic resume. Measured work already saved on disk
will be reused when GPU access is explicitly requested again.

## Accounting and artifacts

Before release, 29 recorded process intervals cover 10.949 overlap-deduplicated
GPU-resident hours across physical GPUs 1–4, including loading and associated CPU
control. This is **not CUDA kernel-active time**, and includes one explicitly
interrupted calibration interval. It must not be reported as 10.949 hours of new
CPU-stage GPU experiments. Burn hours counted: zero.

- [CPU continuation contract](../top_tier_successor_mining/CPU_ONLY_CONTINUATION.md)
- [All-three milestones](../top_tier_successor_mining/BASELINE_MILESTONES.csv)
- [Failure matrix](../top_tier_successor_mining/FAILURE_MODE_MATRIX.csv)
- [Quality evidence table](../top_tier_successor_mining/QUALITY_EFFICIENCY_PARETO.csv)
- [Research scoreboard — unranked](../top_tier_successor_mining/SUCCESSOR_SCOREBOARD.csv)
- [Experiment log](../top_tier_successor_mining/EXPERIMENT_LOG.md)
- [Derived CPU results](../deepep_revalidation/results/top_tier_successor_mining_20260907_135950/analysis/cpu_release_checkpoint/)

Final research decision remains deferred. CPU evidence is not a substitute for
quality-matched request-level performance, cross-model validation, causal tests,
or a successor prototype improving the Pareto frontier.

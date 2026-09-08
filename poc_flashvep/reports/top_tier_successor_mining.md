# Top-tier MoE successor mining — completed GPU resumption

Date: 2026-09-08. **FINAL STATUS: FOUND_INCREMENTAL_ONLY.**
**BEST_SUCCESSOR_CANDIDATE: SERE, selected for falsification; NOT promoted to a paper successor.**

The deferred GPU screening has been executed. None of the three tested avenues
establishes a faithful,material,nontrivial,cross-model successor with demonstrated
quality-matched E2E headroom. This is **not a claim that SERE,Libra or MoDES fail
in their original paper settings**. Existing SERE settings recover most of the
largest measured quality loss; Libra's prediction limitation has little measured
headroom in the valid VL diagnostic; MoDES shows a known quality trade-off plus
a significant deployment-port cost floor.

Kimi and a new successor prototype are **NOT_RUN, conditional gates not met**.
They are not counted as completed cross-model/prototype milestones.

## What was completed after GPU reauthorization

- Only physical GPUs **4,5,6,7** were used. Jobs on0–3 were not stopped.
- Finished the official MoDES1024-question/grid100 frontier, including cached-point
  parity after resumption:436 unique calibration evaluations.
- Official-norm SERE and completed MoDES thresholds on128 ChartQA+128 GQA held-out
  questions; natural greedy output,unchanged official scoring.
- Actual vLLM TP2/DP2/EP4 natural-EOS request comparisons for both tasks at
  offered DP-local cohort caps1/4/16,three randomized paired cohort repetitions.
- Twelve primary clean engine runs contain20,736 request observations including
  the superseded MoDES port. Primary corrected/SERE tables retain16,128 observations.
  These reuse **256 unique questions**,not20K independent tasks.
- Two additional independent GQA-B16 MoDES engine runs with five repetitions each
  test the initially positive but noisy point. These add3,840 request observations.
- Fixed-output,no-op decision-cost,metadata-lifetime,prefill-only and actual
  scheduler/route controls distinguish method behavior from port/measurement cost.
- The supplied native Libra48-layer implementation: text controls,32 captured
  real VL inputs in8 groups,ten paired repetitions/group,exact-current-route
  prediction diagnostic,and a native Nsight overlap trace.
- An additional native text M8192/source control uses32K aggregate tokens and
 20 paired executions. Its no-oracle version passes80/80 first-token comparisons.
  The earlier oracle-interleaved large run is **INVALIDATED**,not speed evidence.

Recorded resumed GPU-resident time: **14.360 GPU-hours**,approximately3.590
four-GPU-equivalent hours. These process intervals include model loading and CPU
control,are overlap-deduplicated per device,and are **not CUDA-active duration**.
Historical+resumed recorded total:25.309 GPU-resident hours. Historical1–4
execution remains historical and is not relabeled as this round's4–7 execution.
Burn contributes **zero** research hours. See GPU_TIME_LOG files and telemetry.

The earlier CPU-release report is archived under
`top_tier_successor_mining/CPU_RELEASE_REPORT_HISTORICAL.md`; its pending statuses
are superseded by this report and BASELINE_MILESTONES.csv.

## Environment and reproducibility boundary

Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`, checkpoint
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
BF16,48 MoE layers,128 routed experts,top8,H2048,I768.
Actual vLLM source path and four-rank runtime logs verify:

- TP2/DP2/EP4,DeepEP high-throughput,`DeepEPHTPrepareAndFinalize`,TritonExperts.
- DBO off,EP enabled,sequence parallel active,prefix cache off,eager path.
- vLLM0.20.0+cu129,torch2.11.0+cu129,DeepEP1.2.1+73b6ea4,
  DeepEP commit73b6ea4a439ba03a695563f9fd242c8e4b02b37c,NCCL2.28.9.
- H10080GB GPUs4–7,driver570.211.01,NV18 connectivity between tested devices.
  Other GPUs/host resources are shared; this is not an exclusive eight-GPU host.

The native Libra environment is separately labeled: author-patched SGLang0.4.10,
torch2.7.1+cu126,CUDA toolkit12.8,Cython planning,and native
AllGather/local-remote/AllReduce execution. It is **not DeepEP**.
Its four DP-attention sources use attention TP1 with four expert ranks,not the
primary vLLM TP2/DP2 topology. The author pool has32 resident+8 replica slots/rank.
The supplied source base is023288645b80fb41b3eed55fd413dd69a7904593;
patch hashes and environment records are retained locally.

SERE official commit8512f39b108cd1fa04e9261a5b6846fb173989e4;
MoDES official933b3ccac3c23e01c763771f1a7d60c4e7ed13a4;
VLMEvalKit d21c5e969983dea6741a98eb2f7ec566ab82ee91.
No irreversible upgrade of the working baseline environment was performed.
Private Libra supplements,model weights,images and full profiling binaries are
not redistributed in the task commit.

## Measurement contract

Primary EP latency is **host submission → engine-core token-ready completion**,
including request rendering/processing,queueing,vision,prefill and natural decode.
It excludes image disk I/O/chat-template preparation and is not HTTP/client
delivery latency. Frontend receipt and polling delay are separately retained.
Token-ready timestamps are copied before mutable runtime stats change.
Coalesced output observations are flagged rather than invented as zero-length ITLs.

Each policy is compared with its own matched vanilla in the same engine and
request cohort. Policies are randomized,with identical warmup and cold image
UUIDs. B1/B4/B16 mean offered cohort cap per DP source,**not constant scheduled
batch size**. Three within-engine repetitions are not three engine restarts.
E2E confidence intervals resample paired cohorts with both DP sources together;
quality intervals resample image clusters. GQA's128 questions contain89 image
clusters; ChartQA has128. A CI containing zero is not an established effect.

HF quality replicas are not EP throughput measurements. Native Libra post-vision
prefill is not full VL request E2E. CUDA stage durations are diagnostic and are
never added across ranks to manufacture request speedups.

## 1. SERE

### Core mechanism and faithful boundary

[SERE](https://arxiv.org/abs/2602.07616) uses the union of current-batch top-S
experts as the primary set,and calibrated expert-response similarity to reroute
secondary slots above rho. Original routing weights remain unchanged; duplicate
slots are not silently merged. It reduces an expert working set,not necessarily
expert FLOPs. Its original single-H20/vLLM0.8.4 performance regime is not EP4.

The400×128 official BF16-normalized calibration and unchanged official CUDA
rerouter were executed. The EP port reconstructs a **DP-local** primary union
across TP sequence shards. An EP-global primary-set design was not tested.
Primitive/sentinel/duplicate-route/no-op controls pass within their stated scope.

### Material observation

Official HF B1 aggressive S2/rho.5 quality deltas:
ChartQA **−11.72pp** [−18.75,−5.47],GQA **−8.59pp** [−14.08,−3.50].
The separate actual EP B1 comparison is even more adverse on ChartQA:

| Actual EP B1 | ChartQA quality Δ | GQA quality Δ | ChartQA request latency change | GQA request latency change |
|---|---:|---:|---:|---:|
| S2/rho.5 |−19.531pp|−7.031pp|+14.808% slower|+8.904% slower|
| S4/rho.5 |0pp|−.781pp|+13.510% slower|+8.845% slower|
| S2/rho.7 |−2.344pp|−1.562pp|+11.875% slower|+8.424% slower|

These are bounded task subsets,not original full-benchmark scores.
Only4 of29 newly wrong HF SERE answers lack EOS; termination/format alone does
not explain the loss. The unchanged scorer remains primary.

### Causal/trivial-fix attack

Existing **S4** recovers **100% ChartQA /88.9% GQA** of the observed aggressive
EP B1 quality loss. Relative to aggressive S2,paired E2E reduction is only
**1.31%** [0.87,1.78] and **.21%** [−.17,1.03],respectively.
This is useful configuration guidance,not a new successor.

With fixed HF B16,the aggressive quality delta is0pp on both tasks.
Among vanilla-correct-at-both-sizes cases,17/17 ChartQA and11/12 GQA B1 failures
recover. Actual online scheduler controls separately show that offered B16
often has only3.5–4 active requests/source during sampled decode.
Batch dependence is an explicit part of SERE,not a newly established modality cause.

The fixed32-output control retains the port slowdown. S8 algorithmic no-op
also costs approximately15.67% in a bounded B16 control,with96/96 through-first-EOS
agreement. Thus the EP port's decision/union cost floor must not be called the
original paper's intrinsic speed loss. Forced post-EOS trajectories are not
universally bitwise identical,including repeated vanilla.

**Headroom:** executed simple-fix quality recovery is large,but no nontrivial
quality-matched additional10% E2E oracle is established. Unknown headroom is
not reported as zero. **Status: INCREMENTAL_ONLY / paper promotion rejected.**
Kimi:NOT_RUN. MLLM-specific causal claim:NOT_ESTABLISHED.

## 2. Libra

### Core mechanism and faithful boundary

[Libra](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9ff1ac9a659085fed0735362cafe5e53-Abstract-Conference.html)
improves the HarMoEny-style prefill execution by using the actual next gate,
prediction-driven expert replication,token sharding,and local/remote overlap.
Incorrect prediction affects locality/planning,not intentionally model math.

The user-supplied native SGLang decoder,lookahead gate,Cython planner,replication
copies,local/remote expert execution and collectives were run,not replaced with
a synthetic DeepEP predictor hook. The VL bridge uses actual captured encoder
outputs,MRoPE and three DeepStack inputs with Qwen-VL weights.

Native vanilla/Libra first-token equality is **320/320** across32 inputs×10
executions. Each matches the HF capture on**31/32 distinct requests**.
One GQA input differs in both native vanilla and Libra; this is an approximate
cross-runtime bridge limitation,not a Libra-specific task error.
Full generation/benchmark quality and continuous VL-serving E2E are not certified.

### Prediction failure vs actual headroom

The CPU matched planner diagnostic found lower Vision recall,but the planner's
locality response was small. The fresh native GPU control then freezes the same
current routes and replaces only lookahead predictions with exact future routes,
retaining predictor GEMM cost and the original planner/streams.

Across eight real VL groups,the median group-paired critical-prefill improvement
is **.146%**,group-bootstrap95%CI **[−.102,1.408]%**.
Group medians range−.312% to1.739%. Excluding the group containing the HF
disagreement gives.184%. This is a **bounded prediction-only diagnostic**,
not a global scheduler oracle or direct request E2E upper bound.

Native VL group medians are87–93ms vanilla versus185–191ms Libra.
The original paper targets235B/355B models on8H200 and explicitly acknowledges
short-context CPU overhead when the local compute window is small.
Missing H100 E40/N768 Triton tuning is also logged. Hence this absolute slowdown
does not establish a novel MLLM failure or reproduce its published speedup.

### Larger-workload and real-overlap checks

Native text M8192/source,32K aggregate,without oracle hooks:
vanilla **377.645ms**,Libra **398.134ms** rank-critical medians;
paired reduction **−5.401%** [−6.045,−5.214]%,
80/80 first-token matches,min logit cosine.999526.
The large oracle-interleaved run failed functional parity and is invalidated.
No large-input exact-prediction speed claim is retained from it.

Nsight on M2048/source confirms genuine copy/expert and NCCL/expert overlap.
Median rank-observation intersections are4.476ms and6.191ms,respectively.
CPU/control/copy costs remain substantial; the trace is consistent with known
short-window overhead. Profiled no-activity time is **not** automatically removable
request time. See LIBRA/NATIVE_OVERLAP_20260908.md.

**Status: NO_MATERIAL_PREDICTION_SUCCESSOR_HEADROOM in the valid tested VL scope.**
Original-scale performance/full-request/generalization remain unverified.
Kimi:NOT_RUN. A better predictor alone is also adjacent to
[PROBE](https://arxiv.org/abs/2602.00509).

## 3. MoDES

### Core mechanism and faithful boundary

[MoDES](https://arxiv.org/abs/2511.15690) is already MLLM-specific: modality/layer
importance from one-layer ablation KL multiplies normalized router weight; calibrated
thresholds skip expert assignments without renormalizing the survivors.

The official1024-question,100-grid frontier completed with436 unique evaluations.
Selected target70/85 thresholds realize71.328%/85.267% calibration assignment
skipping,answer KL.011076/.017023. Calibration and held-out GQA images are disjoint;
1024 questions do not mean1024 distinct images. No held-out label selects thresholds.

The public HF code's fast CUDA implementation remains unreleased. Its simulated
zero-weight path is not claimed as efficient execution. The EP port uses existing
DeepEP invalid-assignment sentinels; exact route/no-op tests validate this boundary.

### Quality and corrected request timing

HF ChartQA quality changes:target70−2.344pp,target85−4.688pp.
The aggressive target85 CI is[−8.594,−.781]pp.
GQA point deltas are−.781pp at both targets,with intervals spanning zero.
The original paper already reports aggressive OCR degradation; it is not a new
modality-aware successor motivation.

The first port wrongly recreated modality metadata per layer. The official
source creates it once per forward. Correcting that lifetime preserved all96
paired natural outputs and improved that port by about9.6%. This is **our port
repair**,not a new MoDES contribution. Only the corrected port is primary below.

| Corrected port | target70 E2E reduction | target85 E2E reduction |
|---|---:|---:|
| ChartQA B1 |−5.019%|−5.539%|
| ChartQA B4 |−5.178%|−5.882%|
| ChartQA B16 |−3.566%|−1.777%|
| GQA B1 |−5.099%|−3.202%|
| GQA B4 |−3.940%|−4.158%|
| GQA B16 initial engine |+7.587%|+1.112%|

Positive means faster. Several B16 intervals span zero. The GQA-B16 initially
positive point was independently challenged:

- target70 three engines:+7.587%,−1.132%,−.909%; across-engine median**−.909%**.
- target85:+1.112%,+.889%,−.358%; median**+.889%**.
- Only one of three target70 runs exceeds5%; no stable large benefit is established.
  Three engines are still a small sample,not a tight universal performance bound.

### Mechanism controls and novelty

Sampled actual B1 MoDES70 decode removes0/768 assignments,while prefill removes
76.79%. At offered B16,the sampled pure-decode removal is3/8,320=.036%;
prefill/mixed is approximately78.4%/77.2%. These are selected-layer/early-step
diagnostics including warmup,not full-workload skip estimates.
The public calibration and paper already distinguish low text/decode skipping.

Algorithmic no-op retains about7.82% port cost in a bounded fixed-output control.
Prefill-only gating preserves the tested64-question ChartQA quality but still
runs about2.85–2.87% slower than stock. This obvious phase option is not a successor.

**Status: KNOWN_QUALITY_TRADEOFF + PORT_COST_LIMITED; no nontrivial successor
headroom established.** Kimi:NOT_RUN. Generic semantic importance/OCR allocation
overlaps [AnyExperts](https://arxiv.org/abs/2511.18314),
[MACS](https://arxiv.org/abs/2605.05225),and recent
[ACE](https://arxiv.org/abs/2609.05228); none of their speedups is imported here.

## All-three ranking and final gate

Scores0–5 assess this project's **successor opportunity**,not paper quality.
Zero for untested cross-model generality means unestablished,not disproved.

| Rank | Base | Score /60 | Main reason |
|---|---|---:|---|
|1|SERE|23|Strongest real quality limitation,but existing S/rho controls repair most of it|
|2|MoDES|20|Known OCR/skip trade-off;corrected EP port has no stable large request benefit|
|3|Libra|18|Valid VL prediction-only headroom is tiny;original-scale/native-port boundaries remain|

Exactly one best candidate was selected:SERE. Its deeper falsification consists
of official calibration,HF/EP transfer,natural membership,fixed length,termination,
no-op cost and existing-knob Pareto controls. It fails the **trivial-fix gate**.
No paper-level failure justifies the conditional Kimi or successor implementation.

See BEST_CANDIDATE_{DEEP_DIVE,CAUSALITY,ORACLE,METHODS,INTRO,PRIOR_ART}.md.
Three hypothetical method principles are documented as rejected/unjustified
branches,not implemented or advertised as new contributions.
The honest seven-sentence Introduction ends with deployment guidance because
the necessary assertion “simple tuning cannot solve it” is false in our data.

## Final recommendation

**FOUND_INCREMENTAL_ONLY. Do not begin a new SERE/Libra/MoDES-successor paper
implementation from these results.**

Keep the reusable baseline ports,official calibration,parity tests,request-level
measurements and negative controls. A future re-opening needs new evidence of a
material residual **after** original knobs and port-cost repair,then quality-matched
E2E headroom and Qwen+Kimi validation. Do not relabel metadata caching,phase gating,
short-context overhead,or lower Vision predictor recall as the new method.

### Remaining limitations — not silently completed

- Original published performance reproduced:NO; model/hardware/runtime regimes differ.
- Full native Libra autoregressive task quality/continuous VL E2E:NOT_VALIDATED.
- Large-input oracle-interleaving correctness:FAILED; invalidated,while no-hook native passes.
- Kimi cross-model successor failure:NOT_TESTED.
- Nontrivial successor feasible E2E oracle:NOT_ESTABLISHED,not fabricated as0%.
- New successor prototype:NOT_RUN,earlier materiality/nontriviality gates not met.
- Broader tasks,long outputs,multi-image/video and cross-engine confidence outside
  the selected MoDES condition remain outside this bounded screen.

## Artifacts

Task:`poc_flashvep/top_tier_successor_mining/`
Results:`poc_flashvep/deepep_revalidation/results/top_tier_successor_mining_20260907_135950/`
Primary tables:`analysis/final_screen_20260908/`
Replications:`analysis/modes_b16_independent_20260908/`
Libra:`analysis/libra_eight_groups_20260908/`,`analysis/libra_native_nsys_20260908/`
Figures:`plots/resume_20260908/`
Resumed accounting:`resume_20260908/GPU_TIME_LOG.csv`

Branch:`flashvep/top-tier-successor-mining`.
Commit/push and final physical4–7 utilize-process verification are recorded in
`DELIVERY.md`/the final handoff. Burn is kept separate from research evidence.

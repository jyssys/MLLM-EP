# Refinement-Aware, EP-Aware Expert Allocation for Diffusion MoE Inference

**PoC specification (research and implementation plan)**  
**Version:** 1.0 — 2026-10-27  
**Model:** `inclusionAI/LLaDA2.0-mini` (the released mini model, **not** `mini-preview`)  
**Required hardware allocation:** CUDA GPUs `0,1,2,3,4,5,6,7`  
**Target distributed configuration:** two independent **physical TRUE EP4** groups (EP4 × 2 replicas); optional isolated single-group EP4 measurements  
**Requested repository destination:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/refinement_aware_ep_expert_allocation_poc_spec.md`  
**Primary objective:** establish whether a *training-free*, diffusion-refinement-aware and *actual EP-cost-aware* token–expert allocation policy can improve **measured request end-to-end latency** while preserving generation quality, compared with strong matched baselines.

> **Execution status:** This document is an experiment specification, not an assertion that experiments have been run. All new measurement fields begin as `NOT_MEASURED`. Do not fill tables with oracle projections or numbers from unrelated environments.

## 0. Executive summary and decision contract

This PoC tests two distinct hypotheses:

- **H1 — Refinement sensitivity:** token–expert edges have heterogeneous marginal importance for the final decoding decision; useful signals are available *causally at decision time* from the current/previous refinement history without training.
- **H2 — EP critical-path value:** among allocation policies with similar retained expert work and generation quality, preferentially removing *low-importance work on the EP critical path* can reduce communication, straggler time, and request E2E latency more than EP-oblivious pruning.

Neither hypothesis is assumed to hold. A third hypothesis, **H3 — complementarity**, tests whether any improvement survives when combined with an Epoch-like live/fresh-token optimization, *only if an implementation is actually available*. An unavailable method is marked `NOT_RUN`; never claim an Epoch comparison from a conceptual proxy.

**Core comparisons:** vanilla TRUE EP4; EP-oblivious same-budget expert pruning; confidence/refinement-only allocation (REFLEX-inspired proxy, with the exact paper algorithm separately evaluated if faithfully implemented); EP-cost-only allocation; joint refinement + EP-cost allocation; and no-op/instrumented EP4. If a cited baseline is unavailable or not faithful, explicitly mark it a `PROXY`, not the original paper.

**Proceed to expensive engineering only if** (a) model and TRUE EP4 baselines pass correctness/quality gates; (b) offline edge sensitivity demonstrates useful quality–work trade-offs; and (c) a calibrated realistic EP cost replay predicts meaningful request-level opportunity *after overhead*. **Primary success target:** >=10% measured median request-E2E reduction against the best matched implemented baseline, with <=0.5 percentage-point GSM8K accuracy decrease; <=1.0 pp is exploratory only. These are research decision thresholds, not promises or guarantees. Always show uncertainty and judge data as paired comparisons. Stop early if estimated opportunity is below 5% E2E after conservative overhead, or if any strong EP-aware advantage over a matched same-work EP-oblivious control disappears.

## 1. Related work and explicit novelty boundary

| Prior work | What it already establishes | Required distinction for this PoC |
|---|---|---|
| REFLEX (arXiv:2608.01784) | Training-free refinement-aware flexible expert computation | Show **measured multi-GPU EP critical-path awareness**, not merely another token confidence or expert-count rule. Compare at equal expert work and quality. |
| Epoch (arXiv:2609.09748) | Block-scoped expert support and sparse fresh-token EP dispatch | Do not relabel selective token computation as new; distinguish *which live token–expert edges* are worth computing under measured rank and communication costs. |
| TEAM and related temporal routing work | Repeated denoising offers routing consistency | No stale-route correctness claims; observe routing but run present-step router. |
| FOCUS / ES-dLLM and other position-wise compute skipping | Concentrating compute on likely decoded positions has precedent | Quantify incremental benefit from expert-edge-level allocation and EP critical-path costs over simpler position-level pruning. |

**Explicitly out of scope in phase 1:** weight training, router fine-tuning, speculative route dispatch, expert relocation/replication, unverified expert-output caching, modifying attention/KV semantics, and oracle information from future steps in an online policy.

### Historical project evidence — contextual, not new results

Previous project records (re-verify source logs in the repository before using in a paper):

- Prior official/HF generation measurements: GSM8K **1211/1319 = 91.81%**, HumanEval **143/164 = 87.20%**. Record the exact run/model SHA, prompting, decoding and scorer before reusing these as a reference. They are *not* fresh measurements and do not automatically reproduce the official benchmark protocol.
- Earlier TRUE EP4 experiments indicated route-edge persistence around **77.68%**, but correction dispatch occurred in around **98.24%** of invocations; this is *not* proof of dispatch overlap speedup.
- Earlier Flash EP4 oracle studies indicated small placement-only E2E headroom (packing ~1.8%, replication <=2.99%, perfect balancing <=5.34%) in that distinct model/workload. Do not transfer those percentages to mini.
- A past TRUE EP4 branch was reportedly `flashvep/dllm-ep-comm-amplification-poc` at commit `dc539d3222193ab0902e0a342679edc3610f54bb`; **verify before reuse**. Do not reset, overwrite, or discard ongoing work.

## 2. Official model and evaluation anchors

The official `inclusionAI/LLaDA2.0-mini` Hugging Face model card (consult as of run date, record a revision/hash) describes ~16B total non-embedding parameters, ~1.4B active per token, 20 layers, and includes an example with `gen_length=512`, `block_length=32`, `steps=32`, `temperature=0.0`, `eos_early_stop=True`. Official *reported* benchmark values: **GSM8K 94.24%** and **HumanEval 86.59%**. These percentages are external references, **not expected guarantees** under a different evaluation harness. The official model card example is not sufficient to establish that its prompt/scoring/decoding protocol reproduces these numbers.

Mandatory links:

1. Model card: https://huggingface.co/inclusionAI/LLaDA2.0-mini
2. Official model family: https://github.com/inclusionAI/LLaDA2.X
3. dInfer: https://github.com/inclusionAI/dInfer
4. REFLEX: https://arxiv.org/abs/2608.01784
5. Epoch: https://arxiv.org/abs/2609.09748

**Do not conflate** `mini` with `mini-preview`, a 16B model with the 100B Flash model, or the official model card's reported score with previously observed local accuracy. Audit router top-k, shared experts, output aggregation weights/normalization, precision, attention implementation, decode strategy and actual layer layout from the pinned checkpoint/runtime, *not by guessing from other MoE architectures*.

### 2.1 Baseline reproducibility gate

1. Freeze an exact checkpoint revision and official tokenizer. Audit chat template, dataset versions, few-shot/zero-shot rules, max generation length, EOS behavior, answer extraction, sampling/threshold/unmask schedule, stopping, seeds and code-evaluation sandbox.
2. Re-run a clean **official HF/reference path** (the one in the model card / closest supported implementation). Capture official settings and current local historical settings as separate rows; do not silently mix them.
3. Re-run `INSTRUMENTED_NOOP` with hooks enabled but all expert edges intact, then vanilla physical EP4 in both groups on a matched deterministic sample set. Validate model outputs and benchmark accuracy; differences due to precision or execution order must be described and tested, not waved away.
4. Conduct a short smoke screen on **calibration/validation** data, then complete GSM8K **1,319-question** test and HumanEval **164-problem** evaluation for reference-baseline and final candidates, if datasets are accessible. Use exact count, correct/total, percentage, **95% Wilson CI**, published reference difference (pp), and exact prompt and scorer ID. HumanEval pass@1 must come from sandboxed execution of completions, not merely textual equality.
5. If the reproduced score is substantially below the official score, perform a **baseline discrepancy investigation** (config/prompt/evaluator/revision) before interpreting method effects. A discrepancy is not automatically a failure if the protocols differ; comparison must be apples-to-apples. Compare proposed variants primarily against a strong **locally reproduced** baseline with *identical* settings.

**Prior local reference summary (not a new run):**

| Benchmark | Official reported `mini` | Prior local HF measurement | Difference, local minus official | New official-config baseline | New physical EP4 |
|---|---:|---:|---:|---|---|
| GSM8K | 94.24% | 1211/1319 = 91.81% | -2.43 pp | NOT_MEASURED | NOT_MEASURED |
| HumanEval pass@1 | 86.59% | 143/164 = 87.20% | +0.61 pp | NOT_MEASURED | NOT_MEASURED |

For historical local values, first verify original artifacts, model revision and evaluation settings before reporting them as established evidence. Round the difference using displayed rounded rates and retain full precision in machine-readable artifacts.

## 3. Eight-GPU topology, process scheduling and hard TRUE EP4 proof

**All eight CUDA devices may be used; target is EP4, not EP8.** Default groups, subject to peer topology verification:

- `GROUP_A`: visible physical GPUs `0,1,2,3`, world size 4, expert-parallel size 4.
- `GROUP_B`: visible physical GPUs `4,5,6,7`, world size 4, expert-parallel size 4.
- Two independent replicas may run simultaneously for independent throughput/quality shards and symmetric A/B comparisons. This is **EP4 × DP2** at the serving-workload level, NOT one EP8 collective.

Before execution capture `nvidia-smi -L`, `nvidia-smi topo -m`, GPU HBM, NVLink/NVSwitch/PCIe connectivity, GPU utilization, CPU memory, `torch.cuda.nccl.version()`, `torch`, CUDA, driver, NCCL, and all local rank→physical GPU mappings. If default groups have poor topology, document and justify an alternate 4+4 partition covering the same eight GPUs, using stable pinned mappings across comparisons.

**TRUE EP4 validation requirements:**

- Show actual expert *weight partition/ownership* by global expert ID and EP rank, real inter-rank dispatch/combine calls and nonzero remote token assignments/bytes; show model-shard inventory/HBM per rank.
- Show an end-to-end equivalence or quality check against reference inference. Real process group world-size 4, EP-size 4 with rank-specific expert placement must be established; **TP4 alone, `device_map=auto`, four independent data replicas, or a simulator do not qualify**.
- Record traffic using real runtime counters or instrumented collective buffers; for proxies derived from token counts, mark `ESTIMATED_BYTES` and never use as measured physical NIC/NVLink traffic.
- Separate useful work from padding, packing, and collective startup; for MoE implementations with all-to-all-v/DeepEP, identify the exact communication primitive and synchronization behavior.
- If TRUE EP4 cannot be achieved on the installed runtime, label `BLOCKED_TRUE_EP4`, report honest partial offline and single-GPU results, and **do not claim EP gains**.

**Eight-GPU utilization plan:**

| Phase | GPUs 0–3 | GPUs 4–7 | Measurement interpretation |
|---|---|---|---|
| Audit/reproduce | HF/reference jobs when practical; TRUE EP4 baseline | Second EP4 baseline and disjoint jobs | Initial throughput and correctness; monitor cross-group contention |
| Trace collection | Vanilla EP4, half of dataset | Vanilla EP4, other half | Merge by immutable sample ID; common frozen configuration |
| Policy/ablation screen | Reference/refinement-only EP4 | Proposed/cost-aware EP4 | Swap groups and repeat (A/B crossover) |
| Final latency | One EP4 group at a time, other group quiescent | Then swap and isolate Group B | **Latency primary results: isolated**, no concurrent load |
| Serving throughput | Vanilla EP4 replica | Another vanilla or proposed EP4 replica | **Aggregate DP2 throughput**, separate from single-request latency |

Do not keep all GPUs busy at the expense of invalid latency measurement. Avoid simultaneous high-bandwidth benchmarks when measuring isolated EP4 latency; test two-group interference as a separately labeled experiment. Shared disk/caching/CPU preprocessing contention must be monitored.

## 4. Dataset partitions, workloads and reproducibility

- **Calibration:** 256 labeled-unneeded or label-hidden GSM8K training prompts (or a distinct approved set), deterministic seed and immutable IDs. Use only model-generated signals; no ground-truth answers to fit the online policy. This is *not* parameter training.
- **Screening validation:** 128 disjoint training/development prompts for feasibility and hyperparameter choice. Freeze all policy thresholds before final test. If a dataset has no approved development partition, create one from a non-test source.
- **Final test:** full GSM8K test (N=1319), HumanEval (N=164) and optionally MGSM/MBPP if feasible and non-overlapping. Exact-prompt matched comparisons. Additional task variety is desirable; record evaluator limitations.
- **Performance workload grid:** batch size `B∈{1,2,4,8,16,32}` where hardware permits, block length `{32,64}` where supported; input and output lengths explicitly enumerated and matched across variants. Use 32/default first. Report OOM rather than dropping a case silently.
- **Decoding defaults for matched historical comparison:** BF16, temperature 0, block length 32, and 32 steps as appropriate to model semantics; prior internal paths sometimes used threshold 0.95. *Do not assume 0.95 is an official requirement:* inspect exact active decoding code and record both the official comparison run and the local legacy comparison run if they differ. Preserve committed token schedule, EOS, and dynamic NFE behavior during an A/B experiment; if a method changes them, report the changes.
- Seed all pseudorandom streams; record sample ordering, model/tokenizer SHA, package lock/version, git commit, command line, environment variables, server load, clock settings, all failed or skipped trials.

**For method selection, do not tune on GSM8K test labels or HumanEval tests.** Report final test once after locking the selected configurations. For statistical confidence use per-question pairs, not per-token as independent samples.

## 5. Mathematical formulation and strict correctness boundary

For model layer `l`, refinement iteration `t`, token position `i`, let the **current** router select `R_{l,t,i}` with aggregation weights `w_{l,t,i,e}`. Baseline routed output (excluding unchanged residual/shared-expert terms) is

`y_{l,t,i} = sum_{e in R_{l,t,i}} w_{l,t,i,e} E_{l,e}(h_{l,t,i})`.

Our training-free policy chooses a subset `S_{l,t,i} ⊆ R_{l,t,i}` from causally available state and observed *physical* EP load/cost estimates. Approximate routed output is

`y'_{l,t,i} = sum_{e in S_{l,t,i}} alpha_{l,t,i,e} E_{l,e}(h_{l,t,i})`,

with `alpha` defined **exactly** for each tested rule: (a) preserve original router weights, or (b) renormalize retained weights. Both are explicit ablations; no accidental weight rescaling. Shared experts, bias, residual connections, attention, layer norms, logits and any mandatory dense paths stay unmodified in phase 1. Enforce `|S| >= 1` unless an explicitly and separately tested zero-edge fallback is valid.

The user-visible **research objective** is to minimize measured request cost at bounded generation-quality degradation. A useful *proxy* is

`min_S  T_hat_EP(S) + lambda * E_hat_decode(S)`

subject to memory, routing, edge-count and permitted quality constraints. `T_hat_EP` is a **calibrated physical EP cost estimator**, not number of experts alone; `E_hat_decode` is a prospective error-risk surrogate computed without future logits/labels. Neither estimate is assumed accurate until validated.

### 5.1 Causal decision-time signals

Candidate signals, included only if cheap and available **before current edge execution**:

- Current router top-k logits/weights, margin between retained/discarded edges, normalized score mass, expert rank/placement.
- Current mask/committed/live status, elapsed iteration, block-relative progress and prior unmask confidence.
- Prior iteration confidence/logit gap for same position (if actually retained by baseline); age since last commit; variation of previous confidence; history validity/version flags.
- Rank-local pending token-expert counts, predicted expert GEMM work, hot rank, peer destinations and expected incremental dispatch/combine costs.
- Optional current pre-MoE hidden norm/statistics and prior iteration *causally cached* features. Avoid any feature requiring full current expert outputs, future router decisions, target labels, post-hoc actual decode selection, or extra dense LM-head calls whose overhead would erase gains.

**A position that looks unimportant may still influence another position through bidirectional attention later.** Hence no assertion that pruned edges are semantically unused; all quality tests must run full autoregressive-in-block/diffusion trajectories, not just teacher-forced isolated layer reconstructions.

### 5.2 Allocation policies (nested ablation ladder)

- `P0_FULL`: current exact top-k router, full routed work. Main quality/correctness reference.
- `P0_HOOKED`: full work through instrumented policy/runtime; isolates instrumentation overhead.
- `P1_UNIFORM`: fixed budget or uniform edge pruning, matched to same total edge count as candidate.
- `P2_ROUTER_ONLY`: keep largest router weights, dynamically or fixed budget, no refinement state or EP costs.
- `P3_REFINEMENT_ONLY`: refinement/confidence + router signals; no rank cost. Include a faithful REFLEX replication if practical; otherwise explicit REFLEX-inspired proxy.
- `P4_EP_ONLY`: same total retained edge count, choose low-impact removals using measured rank costs but **without refinement signal**; do not use future values.
- `P5_JOINT`: refinement + routing importance + critical-path communication/compute estimates. Define deterministic tie-breaking, temperature/threshold handling, and fail-closed bounds.
- `P6_ORACLE_OFFLINE`: hindsight choice using baseline future outcomes and/or true output deltas; **upper-bound diagnostic only**, illegal online. Mark `OFFLINE_ORACLE`, never compare its projected speedup as measured runtime gain.

Test at **matched retained token–expert edges** (e.g., 0%, 5%, 10%, 15%, 20% *actual executed edges removed*), matched candidate-quality regimes, and matched input/output settings. Include confidence floor and critical-rank penalty sweeps selected on validation only. If baseline shared expert persists, report routed-edge saving separately from dense/shared work saving.

### 5.3 Cost-aware policy construction

1. Snapshot the current layer's token→expert→owner-rank assignments *after fresh gate computation*.
2. Estimate per-rank compute and dispatch/combine sizes under the current runtime's padding/packing/expert-kernel shape. Initial simple proxy: `max_r(work_r) + comm_hat`; upgrade to fitted per-rank kernel latency and collective launch estimates.
3. Compute the marginal predicted E2E/critical-path benefit of removing each candidate edge, including its router importance/refinement risk.
4. Apply a deterministic selection rule minimizing risk per marginal critical-path time reduction, under a global work budget and per-token minimum retained expert count. Allow fallback to full top-k if risk is high or cost gain negligible.
5. **Charge all overhead:** policy scoring, masks/prefix sums, compacting, routing metadata, irregular GEMM inefficiency, NCCL startup, barriers and reordering; maintain original token ordering on combine.
6. Validate cost predictions on withheld traces and then using **actual** kernel timings. A policy that removes edges but increases total latency is a negative result, not a success.

Keep a non-oracle fallback for missing history, numerical instability, EOS/tail transitions, unsupported kernel geometry, or an unexpectedly changing block mask. No training or gradient updates; calibration fits only simple timing constants and fixes hand-selected thresholds on calibration/validation sets, not test targets.

## 6. Instrumentation: what must be measured

For every benchmark request, block, iteration, MoE layer and (aggregated) EP rank, record:

- Request/sample ID, dataset split, shape, B, prompt tokens, committed output tokens, block ID, block size, actual NFE and remaining masked tokens.
- Current top-k router ID/weight arrays or privacy-safe counts, live/committed/refresh-required masks, confidence history *with time of availability*, edge-retain mask, executed routed edges, per-position retained k.
- Token-expert demand per rank, expert hotness histogram, per-rank compute workload, rank-load `max/mean`, CV, critical-rank persistence, correlation with refinement status.
- Remote dispatch/combine *logical payload bytes* and if available true collective traffic counters, all-to-all kernel count, bandwidth, packing/padding, actual correction (if any; expected none), memory peak.
- Model forward breakdown: attention, router, policy, pack/scatter, dispatch, expert compute, combine, LM head, decoding/unmask, synchronization, other. Overlapping phases must not be naively summed as disjoint wall time.
- Request wall-clock E2E, per-block E2E, TTFT or time-to-first-committed-token (explicit definition), average forward time, per-iteration NFE latency, steady-state throughput (committed tokens/s), *optional* effective TPOT clearly defined for multi-token diffusion, and wall-clock serving throughput.

**Trace integrity:** instrumented-full and uninstrumented-full should produce the same generations under determinism, except explicitly explained numeric nondeterminism. Quantify instrumentation overhead using matched runs. Do not leak future full-router values into an online decision. Disable heavy tracing for final performance numbers.

## 7. PoC execution phases with fail-fast gates

### Phase 0 — Repository/runtime audit (mandatory)

- Inspect repository/branches/reports, prior TRUE EP4 paths and official model harness. Do not assume a previous path is currently functional. Identify physical expert sharding, source model revision, worker launch, dataset/evaluator, logs and code dependencies. Preserve existing work; create a descriptive new branch when safe.
- Record 8-GPU topology, environment and exact commands to reproduce.
- **Gate:** if only TP4/simulated EP is possible, document `BLOCKED_TRUE_EP4`; report feasibility without inventing TRUE EP performance.

### Phase 1 — Baseline quality and parity (mandatory)

- Official/reference model-quality run, full physical EP4 A and B, no-op hook parity. Audit official GSM8K 94.24% and HumanEval 86.59% comparison; show prompt/decode mismatches when relevant.
- Measure seed-controlled same-sample agreements, masks, NFE, decoded outputs and scorer outputs. Check 95% CIs.
- **Gate:** do not optimize until a defensible, functioning, real EP4 baseline is established. If a local score is below published reference, identify the cause or clearly document a valid protocol difference.

### Phase 2 — Edge sensitivity and refinement mechanism (no runtime changes yet)

- Capture full current-router traces and full-path logits/hidden diagnostics on disjoint calibration/validation data.
- Stratify by iteration, layer, masked vs committed, prior confidence/margin, eventual commitment status (**the last is retrospective analysis only**) and expert/rank hotness.
- Offline perturb one edge or controlled sets; observe current-step logits/decoding effects and **full-trajectory** final quality. Evaluate whether cheap causal proxies predict edge importance.
- Compare naive low-router-weight pruning, confidence-only pruning, and a *retrospective* oracle. State limitations of oracle, including changed future trajectories.
- **Gate:** reject if no causal signal offers a credible quality–expert-work trade-off on validation. Do not proceed because of an oracle-only win.

### Phase 3 — EP cost characterization and trace replay

- Microbenchmark expert compute kernels across realistic per-rank token counts and padding regimes; measure collective latency vs payload and count. Fit and cross-validate `T_hat_EP`, including scheduling and launch overhead.
- Use actual physical placement on both EP4 groups and logged rank work to replay P1–P5 choices. Mark results `TRACE_ESTIMATE`, never actual latency.
- Compute conditional E2E opportunity using measured baseline latency decomposition and conservative overhead; account for overlap and non-additivity.
- **Gate:** if conservative projected request-E2E benefit <5% against strong control, OR incremental EP-aware benefit vs same-work refinement-only <2%, STOP runtime engineering and report NO-GO. If optimism is solely theoretical, mark `INSUFFICIENT_EVIDENCE`.

### Phase 4 — Minimal viable TRUE EP4 implementation

- Implement P1/P3/P4/P5 without modifying model weights or router decisions; optional efficient compact token–expert buffer and count exchange if existing kernels support it. Do not merely set output weights to zero *after* full expert execution—verify actual skipped GEMM and dispatch work.
- Assert per-token minimum k, original router weight semantics, correct gather/scatter restoration, shared experts untouched, no invalid stale cache, exact materialization of current-step state required by decoder.
- Unit tests: retained-all parity, sorted/unsorted indices, B=1 vs large batch, zero-remote assignment, tiny counts, EOS/tail, changing masks, skipped expert, rank metadata conservation, A/B groups parity and CUDA failure handling.
- Keep baseline/full fallback and programmatic fail-closed alarms; measure policy overhead in isolation.

### Phase 5 — Matched quality and measured E2E validation

- **Accuracy first:** full GSM8K test and HumanEval at frozen selected points for P0/P3/P4/P5; at least three independent run seeds if nondeterministic (or report deterministic with one unique trajectory and repeated timing runs). Report count/N, Wilson 95% CI, paired accuracy delta and paired bootstrap CI, changed-answer transitions, exact generation agreement.
- **Performance next:** warmup; randomized variant order or crossover; >=30 measured repetitions per timing condition where feasible; identical batch/sample shapes and output requests; steady clocks; no heavy instrumentation. Repeat in GROUP_A and GROUP_B. Primary latency numbers require *isolated EP4 group* and report p50/p90/p95 along with mean, standard deviation, and per-request paired bootstrap uncertainty.
- Include baseline and proposal at matched total executed routed edges, matched NFE regime, and quality-constrained operating points. If proposals naturally change NFE, report it honestly, and provide an additional matched-NFE controlled comparison where feasible.
- Serving throughput: both EP4 replicas concurrently on distinct sample shards, report aggregate QPS and output TPS, along with each replica's latency under contention. Do not call this EP8 single-request speedup.

### Phase 6 — Generalization, negative controls, and final judgment

- Ablate no refinement/no EP costs/router only/weight normalization/expert budget/critical-path coefficient/layer depth/block length/batch sizes; compare different routing-skew quartiles and quality-hard subsets.
- If original REFLEX or Epoch reference software is practically runnable, include true paper implementation and disclose mismatches, *not* a fabricated numerical baseline. Show complementarity only with identical workload and separately measured components.
- Test non-MoE or single-GPU counterfactual selectively to isolate the EP-specific component; do not confuse algorithmic FLOP reduction with multi-GPU EP gain.
- Publish positive, negative and OOM cases in the same final report.

## 8. Required report templates (populate all fields or `NOT_MEASURED`)

### 8.1 Environment and runtime proof

| Run ID | Git SHA | HF revision | GPU group | Visible physical GPUs | TP | EP | DP replicas | Precision | Router top-k | Actual remote bytes | Evidence path |
|---|---|---|---|---|---:|---:|---:|---|---|---|---|
| TEMPLATE | NOT_MEASURED | NOT_MEASURED | A | 0,1,2,3 | NOT_MEASURED | 4 (VERIFY) | 1 | BF16 (VERIFY) | INSPECT | NOT_MEASURED | NOT_MEASURED |
| TEMPLATE | NOT_MEASURED | NOT_MEASURED | B | 4,5,6,7 | NOT_MEASURED | 4 (VERIFY) | 1 | BF16 (VERIFY) | INSPECT | NOT_MEASURED | NOT_MEASURED |

### 8.2 Model quality and baseline reproducibility

| Dataset | Method | Model revision | Evaluation protocol ID | Correct/N | Accuracy or pass@1 | 95% CI | Official reference | Difference (pp) | NFE mean/p95 | Generation exact-match vs P0 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GSM8K | Published official | model card | PUBLISHED_PROTOCOL_TO_VERIFY | N/A | 94.24% | N/A | 94.24% | 0.00 | N/A | N/A | REFERENCE_ONLY |
| HumanEval | Published official | model card | PUBLISHED_PROTOCOL_TO_VERIFY | N/A | 86.59% | N/A | 86.59% | 0.00 | N/A | N/A | REFERENCE_ONLY |
| GSM8K | Reference HF | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 94.24% | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | PENDING |
| GSM8K | Vanilla TRUE EP4 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 94.24% | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | PENDING |
| GSM8K | Joint P5 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 94.24% | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | PENDING |
| HumanEval | Joint P5 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 86.59% | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | PENDING |

### 8.3 Work reduction and empirical quality–cost Pareto

| Method | Split | Target edge reduction | Actual executed routed-edge reduction | Shared/dense work unchanged? | Mean retained k | Decoding accuracy Δpp | Paired 95% CI | Output agreement | Current/future lookahead? |
|---|---|---|---|---|---|---|---|---|---|
| P0 | validation | 0% | NOT_MEASURED | VERIFY | NOT_MEASURED | 0.00 | N/A | 100% self | NO |
| P3 | validation | 10% | NOT_MEASURED | VERIFY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NO |
| P4 | validation | 10% | NOT_MEASURED | VERIFY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NO |
| P5 | validation | 10% | NOT_MEASURED | VERIFY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NO |
| P6 | validation | OPTIMAL_HINDSIGHT | ESTIMATE_ONLY | VERIFY | ESTIMATE_ONLY | OFFLINE_ONLY | N/A | OFFLINE_ONLY | YES_ORACLE |

### 8.4 TRUE EP traffic, load and communication

| Method | B | Block | Rank max/mean | Rank work CV | Critical-rank ms | Dispatch ms | Combine ms | Expert ms | Remote bytes measured | Collective calls | Padding % | Instrumentation ms |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| P0 | 1 | 32 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| P5 | 1 | 32 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

### 8.5 Measured request latency and throughput (primary table)

| Model | EP group | Mode | Method | Dataset | B | Input tokens | Output tokens | N | Mean E2E ms | p50 E2E ms | p90 E2E ms | p95 E2E ms | TTFT ms | NFE mean | Comm ms | Expert ms | Policy ms | Output TPS | E2E speedup vs best baseline | 95% paired CI | Peak HBM GiB |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaDA2.0-mini | A | ISOLATED_EP4 | P0 | GSM8K | 1 | RECORDED | RECORDED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 1.00x | N/A | NOT_MEASURED |
| LLaDA2.0-mini | B | ISOLATED_EP4 | P5 | GSM8K | 1 | RECORDED | RECORDED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

For speedup, use the correct latency ratio (`baseline E2E / candidate E2E`) on matched requests; report `% latency reduction = 100×(1 - candidate/baseline)` separately. Distinguish mean-of-ratios from ratio-of-means and specify which estimator is primary. **Never calculate throughput speedup by combining unrelated batches.**

### 8.6 Ablation and novelty test

| Variant | Refinement state? | Router scores? | EP cost? | Same actual edge budget? | GSM8K Δpp | E2E speedup | Incremental gain vs P3 | Interpretation |
|---|---|---|---|---|---|---|---|---|
| P1 uniform | No | No | No | Yes | NOT_MEASURED | NOT_MEASURED | N/A | Work-reduction control |
| P2 router only | No | Yes | No | Yes | NOT_MEASURED | NOT_MEASURED | N/A | Router-importance control |
| P3 refinement only | Yes | Yes | No | Yes | NOT_MEASURED | NOT_MEASURED | 0 | Closest algorithmic control |
| P4 EP cost only | No | Yes | Yes | Yes | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | Pure EP signal |
| P5 joint | Yes | Yes | Yes | Yes | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | Claim under test |

### 8.7 Negative results and experiment accounting

| Attempt | Config | GPU-hours | Status (`PASS`/`NO_GO`/`BLOCKED`/`OOM`/`NOT_RUN`) | Reason | Log path |
|---|---|---:|---|---|---|
| TEMPLATE | NOT_MEASURED | NOT_MEASURED | NOT_RUN | NOT_MEASURED | NOT_MEASURED |

## 9. Statistical, benchmarking, and interpretation requirements

1. Report proportion confidence intervals as Wilson intervals on full N. For **paired accuracy difference**, bootstrap by prompt/problem and optionally show McNemar disagreement table; one GSM8K-32 mistake is 3.125 percentage points, so do not infer stable quality from a 32-example screen.
2. Pre-register primary performance metric as **isolated TRUE EP4 request E2E** on matched prompt/output workload; show both absolute time and speedup. Record p50/p90/p95, average and wall-throughput; use multiple timed runs with proper synchronization/warmup and variant order randomization.
3. Keep the official benchmark protocol and speed-benchmark protocol distinct; don't force fixed generation length on an accuracy test if it changes the official decode behavior. Log actual length, stopping, NFE and blocks in all rows.
4. Report exact successful evaluation count `correct/N`, including invalid responses, EOS truncation, parse errors and timeouts; never quietly exclude failures. For HumanEval, isolate untrusted candidate code in a sandbox with resource limits.
5. Every plotted or tabulated data point must be backed by a machine-readable file with sample IDs and command/config hash. Distinguish `MEASURED_GPU`, `MEASURED_WALL`, `TRACE_ESTIMATE`, `OFFLINE_ORACLE`, `PUBLISHED_REFERENCE`, `HISTORICAL_UNVERIFIED`, and `NOT_RUN`.
6. Avoid claim inflation: reduced dispatch bytes alone is not E2E speedup; less expert work alone is not EP-specific novelty; confidence predictability alone is not proof of robustness; absence of a measurable quality drop in a small screen is not losslessness.

## 10. Required repository deliverables

Use descriptive files in the repository. Keep this specification at its required destination. Under `poc_flashvep/reports/` generate:

- `00_runtime_audit.md`: hardware, CUDA versions, branches, model config, environment and dependencies.
- `01_baseline_reproduction.md`: official reported vs verified historical vs newly measured HF/TRUE EP4 accuracy and discrepancies.
- `02_true_ep4_validation.md`: physical rank mapping, actual expert partition, inter-rank evidence, correctness parity.
- `03_refinement_edge_sensitivity.md`: stratified sensitivity, signal availability and validity checks.
- `04_ep_cost_model.md`: measured microbench, fit quality, predictions vs measurements and calibrated overhead.
- `05_oracle_vs_deployable.md`: oracle ceiling, realistic replay and available headroom with clear labels.
- `06_policy_implementation.md`: P1–P5 algorithms, code locations, unit tests and overhead.
- `07_quality_results.md`: full evaluation tables (correct/N, 95% CI, deltas and mismatches).
- `08_latency_ep_results.md`: isolated E2E breakdown, A/B crossover, throughput and resource tables.
- `09_ablations.md`: equal-budget and equal-quality comparisons, matched NFE cases.
- `10_prior_work_comparison.md`: faithful implementations, proxies and unavailable methods explicitly separated.
- `11_final_decision.md`: `GO`, `CONDITIONAL`, `NO_GO` or `BLOCKED`, supporting measurements and next best test.

Create structured artifacts in `poc_flashvep/results/refinement_aware_ep_expert_allocation/` (or compatible existing layout): frozen `run_config.json`, `environment.json`, `baseline_accuracy.csv`, `quality_samples.jsonl`, `ep_rank_metrics.csv`, `refinement_edges.parquet` (when feasible), `latency_samples.csv`, `policy_ablation.csv`, and command logs. Output repeatable run scripts/CLI entrypoints and a concise `README` with exact launch commands. Avoid storing huge token/hidden dumps unless needed; aggregate and sample, with storage caps. Redact credentials and sensitive paths from logs.

Every report must identify runtime and model versions, source of figures, success/failure states, missing values, and be written in **English**. Markdown tables should not be truncated. Do not copy arbitrary numbers from this planning document into new experiment-result columns.

## 11. GO / NO-GO decision tree

| Checkpoint | Necessary evidence | Decision |
|---|---|---|
| A — baseline | New correct/total, CI; reference protocol audit; TRUE EP4 sharding + traffic; no-op parity | Stop as `BLOCKED_BASELINE` if impossible |
| B — causal importance | Held-out validation quality–work curve materially better than naive uniform/router pruning | `NO_GO_ALGORITHM` if absent |
| C — EP headroom | Calibrated predicted marginal EP value exceeds implementation costs; conservative >=5% request E2E projected | `NO_GO_HEADROOM` if below |
| D — EP specificity | Joint P5 outperforms matched-edge P3 in *measured* E2E by target >=2% incremental; show CI | `NO_GO_EP_NOVELTY` if absent |
| E — full validation | Full GSM8K + HumanEval, ideally <=0.5 pp score drop and >=10% measured isolated EP4 E2E reduction | `GO` if robust; <=1 pp exploratory may be `CONDITIONAL` |
| F — robustness | Batch/block/length/seed/hardware group crossover, no hidden OOM or quality regressions | Downgrade to `CONDITIONAL` or `NO_GO` if fragile |

The final conclusion must explicitly say whether the PoC establishes (i) algorithmic benefit, (ii) physical EP-specific benefit **beyond work-count reduction**, and (iii) a plausible novel ML contribution relative to REFLEX/Epoch/FOCUS. If only trace-level potential is demonstrated, say so.

## 12. Agent execution constraints and acceptance checklist

- [ ] Read this entire specification and verify exact run-time repo contents. Do not infer functionality from stale filenames.
- [ ] Use only GPUs 0–7; configure two disjoint physical 4-GPU EP groups and prove **TRUE EP4**.
- [ ] Verify checkpoint `inclusionAI/LLaDA2.0-mini`, non-preview variant and exact model revision.
- [ ] First reproduce the vanilla model's absolute GSM8K/HumanEval quality as closely as practicable under an auditable scorer/protocol; report official vs historical vs new measurements honestly.
- [ ] Implement causal policies P0–P5 and exclude all hindsight from online runtime. Re-use current router, unchanged model weights, attention and shared experts.
- [ ] Show actual skipped collective payload/expert kernel work, not just algebraically masked outputs.
- [ ] Compare at equal work, equal input shapes, similar quality, and when possible matched NFE.
- [ ] Use both EP4 groups efficiently for screening and throughput, but isolate groups for primary E2E latency and perform A/B crossover.
- [ ] Fully populate reproducible English reports and per-run CSV/JSONL; all omitted or blocked cases explicitly labeled.
- [ ] Never present estimated/oracle gains as achieved acceleration, paper scores as local reproduction, or EP4×2 as EP8.
- [ ] Deliver a **clear measured-data-backed GO/NO-GO**, even if negative, without adding unjustified downstream phases.

---

**Reference verification date:** 2026-10-27. This file was prepared as a PoC plan; no remote-machine operations, TRUE EP tests, accuracy runs, or latency measurements were performed by the document author. The execution agent must create the actual repository version at the destination listed above and independently verify every historical number and source protocol.

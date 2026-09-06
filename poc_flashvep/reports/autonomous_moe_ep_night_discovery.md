# Autonomous MoE-EP Night Discovery

## Executive judgement

**FINAL STATUS: `SEARCH_SPACE_EXHAUSTED_NO_GO`**

This sprint deliberately searched for a high-mass, causal, non-trivial
phenomenon in real Qwen3-VL MoE expert-parallel serving.  Forty-nine candidate
hypotheses were catalogued; more than twenty were tested with fresh live
4-GPU serving data, including a final 3,600-second randomized worker campaign.
No candidate survived the direct request-level effect, causal-control, and
generality gates required for a paper-level strong direction.  The strongest
signals were measurement warnings (wave completion spread and common runtime
state transitions), not removable MoE work.

The result is intentionally negative: the tested TP2/DP2/EP4 DeepEP
high-throughput path does not expose a stable multi-percent optimization
opportunity in the investigated dimensions.  Future work should move to a
different causal variable or workload, rather than rename these closed
directions.

## Reproducibility envelope

- Repository: `/home/esjung/MLLM-EP-github`
- Branch: `flashvep/autonomous-moe-ep-night-discovery`
- GPUs: physical 1,2,3,4 only (`CUDA_VISIBLE_DEVICES=1,2,3,4`), four H100
  HBM3 devices; PCI buses 08:00, 0D:00, 12:00, 17:00.
- Model: Qwen3-VL-30B-A3B-Instruct, snapshot
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Configuration: BF16, TP2/DP2/EP4/PP1, DeepEP high-throughput,
  `DeepEPHTAll2AllManager`/`DeepEPHTPrepareAndFinalize`, eager, DBO off,
  prefix cache off, async scheduling default.
- Software: vLLM 0.20.0+cu129, torch 2.11.0+cu129, DeepEP 1.2.1+73b6ea4,
  flashinfer 0.6.8.post1, NCCL 2.28.9, driver 570.211.01.
- Model configuration independently records 48 MoE layers, 128 routed
  experts, top-8 routing, hidden size 2048 and expert intermediate size 768.
- Session wall interval was approximately 6 h 57 min; valid live serving
  accumulated 18,241.93 s (5.07 h) or 20.2689 four-GPU-hours.  The invalid
  early H04 attempt is excluded.  `GPU_TIME_LOG.csv` is the accounting source.

Three source-discovery passes read the local vLLM 0.20 worker, DP coordinator,
GPU runner, scheduler and DeepEP prepare/finalize implementation.  Notable
facts were recorded before choosing experiments: eager DP still performs a
per-step CPU process-group rendezvous; DeepEP dispatch returns asynchronous
dependencies; metadata is rebuilt from a host list and copied to the device at
each MoE layer; and dummy-batch RPCs are real worker work.  These were measured
as controls, not assumed improvements.

## Measurement protocol

The inherited observer was first audited against a no-hook server.  Buffered,
deferred same-device CUDA events and sampled layers 0/12/24/36/47 avoid a
per-layer synchronize and route copy, but the observer still added 3.27% E2E
median overhead (TTFT +1.62%) in the H01 trust test.  Consequently, absolute
cross-run numbers are treated cautiously and randomized A/B comparisons within
one worker are primary.  Each final block used warmup, randomized variant
order, successful HTTP requests, and paired request-wave summaries.  TP rows
were reduced to one logical DP step; no cross-device absolute CUDA timestamp
subtraction was performed.

The final v7 live campaign completed 21,456 successful request rows and 59,789
logical sampled steps (298,945 sampled layer rows) across H38/H43/H33/H45/H46/H48.
The fresh analyzer and diagnostics are under the result root; plots are
generated from the same CSVs.

## Findings with direct evidence

### 1. Wave completion spread is not request latency

H06/H37/H46 repeatedly produced large last-completion/wave differences (about
86–92%, and H48 about 17%) while the per-request median was near zero to a few
percent.  Fresh v7 H46 measured wave B-vs-A +89.62% but request p50 -2.30%;
H48 measured wave -17.13% but request p50 -0.29%.  These are different output
budgets/cohort completion semantics, not an MoE optimization opportunity.

### 2. Common runtime regimes move both treatments together

H09/H14 first exposed large common transitions.  H45 fresh replication had
562/561 steady waves, request p50 -0.12%, while both A and B shared a high
coefficient-of-variation wave regime.  H46 moved together from roughly
3.4/6.5 s to 5/9 s, and H48 started in the same elevated state.  This is a
useful state-drift warning, but no treatment-specific causal lever was found.

### 3. Host metadata construction is frequent but normally tiny

The direct `ExpertTokensMetadata.make_from_list` observer saw 287,219 calls:
median 0.1137 ms, p90 0.125 ms, p99 0.145 ms, maximum 13.9 ms.  H33 fresh
request p50 was +1.73%, but its A/B output budgets differed and wave p50 split
91.78%; H38 likewise used A=1 versus B=16 output tokens.  The data do not
identify metadata allocation as a removable request-level mass.

### 4. Idle-DP participation is real, not valuable in this workload

The corrected worker hook observed 6,399 `execute_dummy_batch` calls per DP1
worker (286 per DP0 worker) in H36.  The useful-request paired effect was only
-0.26% (H36), so dummy participation is confirmed but not a high-mass lever.

### 5. Async/DP backend controls are null

H31 with `--no-async-scheduling` kept DeepEP HT active and changed request p50
by -1.83% (wave -1.90%).  H32 with NCCL DP synchronization forced
(`disable_nccl_for_dp_synchronization: False`) changed request p50 by -0.03%
and wave by -0.01%.  H43 pinning versus unpinned DP was -0.15% at request p50.
No simple scheduler or rendezvous flag explains the prior anomaly.

### 6. Attention/MoE co-tail is not stable evidence of an EP-only effect

The earlier deferred trace showed attention/MoE p95 co-tail enrichment of about
14.2x, motivating a generic whole-GPU control.  The independent fresh v7
campaign instead measured Spearman 0.038 and simultaneous tail overlap 0.0407%
versus 0.25% independence expectation.  The discrepancy makes this a
protocol/state-sensitive diagnostic, not a reliable EP-specific candidate.

### 7. Vision effects are front-end effects, not a new MoE method

H15 image versus text predecessor showed an early roughly 9% request effect
that collapsed to 1–2% after state and shape controls.  H23 one-image versus
two-image requests were about 0.3%.  Existing vision/DeepEP overlap negatives
remain: dispatch -12.4%, combine -5.0%, expert -8.9% under the prior bounded
test.  No new multimodal causal variable was found.

### 8. Layer and age effects are weak

Fresh v7 sampled MoE p99 was approximately 2.91–2.97 ms across layers 0, 12,
24, 36 and 47.  Attention/MoE layer correlations were 0.113 at layer 0 and
between -0.016 and -0.003 at later sampled layers.  H27 late-vs-early decode
age was +0.84% median with high p90 absolute variability, not a stable lever.

## Hypothesis coverage

The catalog contains 49 causal nodes (H01–H49, with aliases explicitly marked).
The live campaign directly exercised H01, H03–H12, H14–H15, H23–H25, H28,
H31–H33, H36–H38, H40, H43, H45–H48; source and diagnostic passes covered
the remaining nodes or closed them as unsupported/low-priority.  The final
scoreboard preserves status, effect, repetitions, causal clarity, generality,
novelty risk and trivial-fix assessment.

| Candidate family | Representative fresh result | Judgement |
|---|---:|---|
| DP partition/phase/heterogeneity (H03–H05,H07) | request effects about -1.8% to +0.2% | NO_GO |
| Output churn/completion spread (H06,H37,H46,H47) | wave +87–92%, request median -2.3% to +0.1% | NO_GO / metric caution |
| Shape/context/order (H08–H11,H14,H26) | no stable request-level multi-percent effect | NO_GO |
| Arrival/history/turnover (H12,H25,H28,H45,H48) | common-state or <=2.5% request effects | NO_GO |
| Vision/image composition (H15,H23) | ~0.3–2% after controls | NO_GO |
| Metadata host path (H33,H38) | 0.11 ms typical; output-confounded wave split | NO_GO |
| Async scheduler / DP backend (H31,H32,H43) | -1.83%, -0.03%, -0.15% | NO_GO |
| Whole-GPU co-tail (H24,H40) | protocol-sensitive, no robust E2E mass | HOLD diagnostic / no candidate |

## Rethink checkpoints and autonomous expansion

`RETHINK_5.md`, `RETHINK_10.md`, `RETHINK_15.md` and `RETHINK_20.md` record
the forced assumption audits and generated children.  The key correction was
to stop treating a sampled MoE or wave span as automatically user-visible
latency.  `ANOMALIES.md` records the common regime transitions; `SURPRISE_LOG.md`
records the observer tax, CPU rendezvous, metadata cadence and dummy-RPC
observability.  The active frontier was frozen only after the final v7
replications.

## Research ranking

Scores use the requested 1–5 dimensions (effect strength, robustness, causal
clarity, novelty potential and implementability are summarized below; the
full per-node table is in `CANDIDATE_SCOREBOARD.csv`).

1. **Common serving-state regime transitions** — effect strength 4, robustness
   3, causal clarity 2, novelty 3, implementability 2.  Surprising and
   measurable, but no intervention-specific E2E mass; generic runtime/state
   risk is high.
2. **Wave completion spread versus request semantics** — 4, 4, 4, 2, 3.
   Highly reproducible and important for evaluating MoE serving experiments,
   but adjacent to ordinary batching/API completion behavior.
3. **Per-layer metadata host cadence** — 2, 3, 2, 3, 3.  A real source-level
   assumption with rare outliers, but measured mass is too small and the first
   causal contrast was output-confounded.
4. **Attention/MoE co-tail diagnostic** — 3, 2, 2, 2, 2.  Worth a future
   whole-GPU observability study only if a low-perturbation trace resolves the
   protocol discrepancy.
5. **Idle DP dummy work** — 2, 4, 4, 2, 3.  Directly observed, yet request
   effect is null in the validated workload.

### Best research direction

If a new project is desired, the cheapest next direction is a **whole-serving
state-observability study**: explain common regime transitions using clock,
allocator, KV/workspace and host-gap telemetry with a no-hook or lower-overhead
trace.  It is not yet an MoE optimization paper and should be promoted only if
the state can be causally perturbed and carries recurring request-level mass.

### Second-best direction

Build a benchmark methodology paper around **completion-spread versus
request-critical latency** in continuous batching.  It is systems-useful but
prior-art risk is high and it is not an EP-specific method.

### Do not pursue in this branch

Do not implement idle-DP balancing, a simple async-scheduling toggle, NCCL DP
backend switching, metadata reuse, wave-only completion optimization, image
count policy, fanout-aware routing, modality TP/EP switching, coalescing,
fragmentation laws, selective synchronization, RL, or a production scheduler.

## Direct answers to generality questions

- **Dense model:** not tested in this resumed campaign; no generic dense claim
  is made.
- **MoE without EP:** not tested; the observed nulls do not establish that
  distributed execution is irrelevant.
- **EP interaction required:** source and DeepEP controls show distributed
  dependencies are present, but no high-mass causal intervention survived.
- **MLLM-specific:** no surviving candidate requires modality; observed image
  effects are front-end or workload-shape effects.

## Artifacts

- Working contract: `poc_flashvep/reports/autonomous_moe_ep_night_discovery_spec.md`
- Source/queue/checkpoints: `poc_flashvep/autonomous_moe_ep_night_discovery/`
- Primary result root:
  `poc_flashvep/deepep_revalidation/results/autonomous_moe_ep_night_discovery_20260907_000146/`
- Fresh v7 trace: `live_v7/`; parsed evidence: `analysis/`; figures: `plots/`.
- Machine-readable manifest: `EXPERIMENT_MANIFEST.json`.

The result is a conservative closure of the searched space, not a claim that
all MoE serving research is exhausted.  A future candidate should start from
a new causal variable and pass the same direct E2E headroom gate before any
method implementation.

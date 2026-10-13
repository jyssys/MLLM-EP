# MultiBD × MoE Expert Parallelism PoC
## Does Multi-Block Diffusion Parallelism Translate into Physical EP Throughput?

**Target repository:** `/home/esjung/MLLM-EP-github`  
**Recommended server path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/multibd_moe_ep_poc_spec.md`  
**Available GPUs:** `2,3,6,7` only  
**Primary hardware target:** TRUE physical EP4  
**Primary serving regime:** batch-1 / low-request decoding  
**Primary model family:** MBD-LLaDA2-Mini (MoE)  
**Primary workload:** coding with the official MBD-Code-LLaDA2-mini checkpoint

---

## 1. Research Question

Multi-Block Diffusion (MultiBD) increases decoding parallelism by keeping multiple consecutive noisy blocks active in a bounded running-set.

The key systems question is:

> Does the algorithmic parallelism gained by MultiBD still translate into wall-clock throughput when the underlying dLLM is a sparse MoE executed with physical Expert Parallelism?

Two opposing effects are possible.

### Positive effect
- more live tokens per forward;
- more tokens per active expert;
- better expert GEMM utilization;
- better amortization of launch/collective overhead;
- higher Tokens Per Forward (TPF).

### Negative effect
- larger active-expert union;
- larger destination-rank fanout;
- denser EP communication graph;
- larger Dispatch/Combine payload and latency;
- larger max-rank expert load / straggler severity;
- step latency increasing faster than useful-token progress.

The PoC should determine which effect dominates.

---

## 2. Conceptual Clarification

MultiBD is **not a MoE architecture**.

```text
MultiBD / SingleBD = decoding/execution regime
Dense / MoE        = model architecture
EP                 = multi-GPU execution strategy for MoE experts
```

The canonical MBD-LM approach is also **not purely training-free**. Direct MultiBD inference on a SingleBD-trained model is possible, but the MBD-LM work reports a train–inference mismatch and quality degradation. It therefore introduces Multi-block Teacher Forcing (MultiTF) post-training.

The public MBD-LLaDA2-Mini checkpoints are post-trained MBD models built on the MoE LLaDA2-Mini backbone.

Therefore, the primary PoC should use an official MBD-LLaDA2-Mini checkpoint rather than treating MultiBD as a training-free modification of the original LLaDA2.0-mini.

---

## 3. Primary Fair Comparison

Keep the **same MBD checkpoint and same weights** and vary only the number of concurrently active block slots.

Use the official Diffulex semantics:

```text
Buffer/running-set size = 1  -> SingleBD-equivalent
Buffer/running-set size >= 2 -> MultiBD
```

Primary sweep:

```text
R = 1, 2, 3, 4
```

Do not compare different model weights and attribute the difference purely to MultiBD.

An optional native LLaDA2.0-mini SingleBD checkpoint may be shown separately as context, but it is not the primary systems baseline.

---

## 4. Model Selection

### Primary

Use the non-DMax code checkpoint from the official release, preferably:

```text
SJTU-DENG-Lab/MBD-Code-LLaDA2-mini-16B
```

Reasons:
- LLaDA2-Mini is MoE;
- checkpoint is MultiTF-post-trained for MultiBD;
- coding was communication-exposed in prior physical EP4 LLaDA2 experiments;
- HumanEval+/MBPP+ are official MBD evaluation domains.

Do **not** use DMax in the primary PoC because it adds another decoding mechanism.

### Secondary

If cheap, use the corresponding math checkpoint on GSM8K/MATH500.

Do not pool different checkpoints without labeling them.

---

## 5. Runtime Selection

Prefer the official experiment-reproduction path:

```text
Diffulex `mbd-lms` branch
```

for the MultiBD state machine and Block Buffer behavior.

The decisive result requires **real physical EP4**, not an offline expert-to-rank projection.

Stage 0 must inspect whether the chosen Diffulex/vLLM stack already exposes a valid physical EP path for the LLaDA2 MoE checkpoint.

If a small bounded integration is needed, reuse existing EP primitives and prior project instrumentation. Useful prior project assets include:

```text
poc_dllm_ep_comm
poc_dllm_block_local_replica
poc_dllm_attn_moe_pipeline
```

Reuse:
- expert ownership verification;
- router top-k tracing;
- destination-rank mapping;
- destination-deduplicated D+C accounting;
- per-rank load statistics;
- CUDA-event stage timing.

Do not rewrite a serving engine. If TRUE physical EP4 cannot be established with bounded effort, report:

```text
RUNTIME_BLOCKED
```

---

## 6. Hardware

Use only:

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7
```

Record:

```bash
nvidia-smi
nvidia-smi topo -m
```

Save physical UUIDs, logical-rank mapping, expert ownership, topology, free HBM, and software versions.

Do not terminate other users' processes.

---

## 7. Prior / Novelty Boundary

### MBD-LMs
Already provides MultiBD, MultiTF, bounded running-set, Block Buffer, decode-store overlap, static-shape execution, and reported TP2 throughput.

Do not claim MultiBD itself as a contribution.

### Epoch
Already optimizes sparse-MoE dLLM execution within a diffusion block by reducing redundant expert work and EP traffic.

Do not claim decoded-token skipping or active-expert support compilation.

The new axis is **inter-block concurrency**: multiple noisy blocks are active simultaneously.

### TEAM
Already addresses the many-experts / few-accepted-tokens mismatch in MoE dLLMs.

Do not claim generic expert-activation reduction.

### BlockServe
Already addresses block-grained scheduling and convergence heterogeneity across requests.

This PoC is different:

```text
one request
+
multiple dependent noisy blocks active concurrently
+
physical MoE EP
```

Create `prior_overlap.md` before finalizing.

---

## 8. Stage 0 — Validate MultiBD Before EP Analysis

Before EP-specific conclusions:

1. load the official MBD checkpoint;
2. verify SingleBD-equivalent `R=1`;
3. verify MultiBD `R>=2`;
4. verify active block/slot transitions;
5. measure basic TPF on a few probes;
6. verify quality is sane.

If feasible, reproduce only a small subset of the paper's non-EP/TP behavior to validate the state machine.

Do not spend the PoC reproducing the full paper.

---

## 9. Stage 1 — TRUE EP4 Validation

On GPUs 2,3,6,7 verify:

- expert weights are physically sharded;
- expert ownership matches routing;
- real Dispatch and Combine collectives occur;
- owner-local expert execution occurs;
- all four ranks participate;
- no silent replicated-expert / TP-only fallback is used.

Save a small proof trace.

This gate must pass before any EP-specific research conclusion.

---

## 10. Primary Workload

Primary:

```text
HumanEval+ and/or MBPP+
```

using the code MBD checkpoint.

Suggested discovery cohort:

```text
8 calibration problems
24 held-out problems
```

or a comparable frozen subset.

Secondary if cheap:

```text
GSM8K / MATH500
```

with the math MBD checkpoint.

Freeze samples before observing routing/latency results.

---

## 11. Accuracy / Correctness

R=1 and R>1 may legitimately produce different token sequences, so exact token parity **between different R values is not required**.

Measure quality using official benchmark metrics:

```text
HumanEval+ / MBPP+ pass@1
GSM8K exact match
MATH500 official accuracy
```

However, instrumentation parity is mandatory:

```text
same R, tracing OFF
vs
same R, tracing ON
```

must preserve generated tokens, forward/NFE counts, and observable acceptance behavior.

Do not report truncated outputs as benchmark accuracy.

---

## 12. Running-Set Sweep

Primary sweep:

```text
R = 1, 2, 3, 4
```

If the runtime exposes Block Buffer size rather than a literal running-set size, record the exact official parameter and its mapping to active blocks.

Do not tune R per request in the primary comparison.

Use paired requests across all R settings and counterbalance execution order where practical.

---

## 13. Core Algorithmic Metrics

For every request/R:

```text
model forwards
useful committed tokens
TPF = committed tokens / forward
step latency / forward
TPS = committed output tokens / second
request E2E generation latency
```

Use actual commit semantics from the runtime.

---

## 14. Core MoE / EP Metrics

For every forward, sparse layer, and rank record:

```text
active blocks / slots
live token count
newly committed tokens
active experts
active expert union
expert assignments
tokens per active expert
expert-token histogram
source->destination rank edges
destination-rank fanout
rank-edge density
Dispatch bytes
Combine bytes
Dispatch ms
Expert ms
Combine ms
max-rank expert assignments
max/mean rank load
rank CV
hottest rank
```

Use actual backend bytes if available; otherwise use validated destination-deduplicated logical payload bytes and label them clearly.

Do not treat expert-assignment count as physical bytes.

---

## 15. Key Derived Metrics

### 15.1 EP Cost per Useful Token

```text
EP_time_per_commit
=
(Dispatch_ms + Expert_ms + Combine_ms)
/
newly committed tokens
```

Also:

```text
D+C_bytes_per_commit
=
(Dispatch_bytes + Combine_bytes)
/
committed tokens
```

### 15.2 Parallelism Conversion Efficiency (PCE)

For each R relative to R=1:

```text
TPF_gain(R) = TPF(R) / TPF(1)
TPS_gain(R) = TPS(R) / TPS(1)
PCE(R) = TPS_gain(R) / TPF_gain(R)
```

Interpretation:

```text
PCE ~ 1.0 -> algorithmic TPF gain translates well to throughput
PCE << 1  -> added parallelism is being lost to higher step cost
```

### 15.3 EP Incremental Step-Latency Share

For R>1:

```text
Delta_step = StepLatency(R) - StepLatency(1)
Delta_EP   = (D+C+Expert)(R) - (D+C+Expert)(1)
EP_Incremental_Share = Delta_EP / Delta_step
```

Use paired request quantities. If Delta_step <= 0, report separately.

### 15.4 Expert Working-Set Expansion

```text
WS_expansion(R)
=
active expert union(R)
/
active expert union(R=1)
```

Also report tokens per active expert.

### 15.5 Rank-Graph Pressure

Report rank-edge density, active destinations, max/mean load, CV, and hottest-rank load.

If R=1 is already fully dense, do not claim graph densification; focus on payload/load/timing instead.

---

## 16. Strong EP-Specific Control

If feasible with the same checkpoint/runtime/hardware, run a non-EP distributed baseline:

```text
TP-only / EP disabled
```

or the strongest fair supported non-EP configuration.

Sweep the same R values.

Compare:

```text
PCE under TRUE EP4
vs
PCE under TP-only
```

and step-latency scaling.

This control is extremely valuable because it separates generic MultiBD overhead from EP-specific conversion loss.

If fair TP-only execution is unsupported, document why rather than fabricating it.

---

## 17. Main Outcomes

### Outcome A — EP scales well

```text
TPF increases
TPS increases proportionally
EP_time_per_commit stays flat or decreases
PCE stays high
```

Verdict: `NO-GO`

### Outcome B — Generic MultiBD overhead

```text
step latency rises
but EP does not explain it
TP-only and EP4 degrade similarly
```

Verdict: `GENERIC_MULTIBD_ONLY`

### Outcome C — EP-specific conversion loss

```text
TPF rises substantially
TPS gain is much smaller
EP D+C/expert/rank pressure explains a large fraction of added step latency
and/or EP cost per committed token worsens
```

Verdict: `GO` or `CONDITIONAL` depending on robustness.

---

## 18. Discovery Gates

### Gate 1 — Algorithmic Parallelism

At some R>1:

```text
TPF_gain >= 1.30x
```

### Gate 2 — Poor Parallelism Conversion

Interesting:

```text
PCE <= 0.75
```

Strong:

```text
PCE <= 0.65
```

If PCE >=0.85 and TPS scales well, likely NO-GO.

### Gate 3 — EP Explains the Loss

Desired:

```text
EP Incremental Step-Latency Share >= 50%
```

Alternative strong signal when TP-only exists:

```text
PCE_EP4 at least 15 percentage points worse than PCE_TP-only
```

### Gate 4 — EP Cost per Commit

GO signal if increasing R causes either:

```text
EP_time_per_commit >= 20% worse
```

or:

```text
D+C_bytes_per_commit >= 25% worse
```

while TPF rises.

### Gate 5 — Physical Bottleneck Magnitude

Prefer:

```text
D+C+Expert >= 30% of step latency
```

at the problematic R.

### Gate 6 — Robustness

Effect should repeat across at least two R>1 settings and a meaningful fraction of held-out requests, or across code+math if secondary evaluation is run.

---

## 19. Quality Gate

The problematic MultiBD setting must remain meaningful as a model configuration.

Prefer quality within roughly 1–2 percentage points of the R=1 same-checkpoint baseline, while also reporting the official MBD model's known quality behavior.

Do not promote an R that only achieves high TPF through severe accuracy loss.

---

## 20. Diagnosis if EP-Specific Loss Exists

Decompose the cause.

### A. Expert working-set expansion
Does active-expert union grow faster than tokens/expert?

### B. Communication payload
Do D+C bytes grow disproportionately?

### C. Rank imbalance
Does max-rank load grow or become more variable?

### D. Graph density
Does MultiBD touch more rank edges?

### E. Positive batching effect
Do larger tokens/expert improve kernel utilization enough to offset A–D?

The final report must identify which effect dominates.

---

## 21. Optional Slot / Noise-State Analysis

If exposed by the runtime, record per active slot:

```text
slot index / block age
mask ratio or live-token count
newly committed tokens
expert distribution
rank distribution
```

Ask whether high-noise and low-noise concurrent blocks create different EP pressure.

This is secondary and should not delay the core PoC.

---

## 22. Do Not Implement the Final Method Yet

Do not implement yet:

- adaptive running-set controller;
- learned EP-cost predictor;
- expert replication/remapping;
- altered router;
- approximation;
- new CUDA kernel;
- modified MultiTF training.

First establish whether fixed MultiBD actually creates an EP-specific conversion problem.

---

## 23. If GO: Smallest Next Experiment

Only if the pathology is real, the next candidate is:

```text
EP-aware MultiBD running-set admission/control
```

Use current observable state first:

```text
live tokens
active expert union
tokens per active expert
rank load / CV
destination fanout
D+C pressure
slot noise state
```

Test an oracle/simple threshold controller before learned policies.

Avoid future-routing prediction in the first method.

---

## 24. Timing Methodology

Use separate lanes:

```text
clean E2E
event-only timing
heavy routing trace
```

Do not use heavy trace wall time as headline latency.

Prefer event observer overhead <=5%. If >10%, mark fine-grained stage attribution as untrusted and retain clean E2E results only.

---

## 25. Statistics

Requests/problems are independent units.

Do not treat forwards/layers/blocks/experts as independent samples.

Use paired request-level bootstrap CIs (>=2000 resamples if cheap) for:

```text
TPF gain
step-latency ratio
TPS gain
PCE
EP_time_per_commit change
D+C bytes/commit change
EP incremental share
E2E/TPS
accuracy difference
```

---

## 26. Required Plots

1. **R vs TPF / step latency / TPS**
2. **R vs PCE**
3. **R vs EP ms per committed token / D+C bytes per committed token**
4. **R vs active expert union / tokens per active expert**
5. **R vs rank-edge density / max-mean / CV**
6. **R-specific latency breakdown**
7. **Quality vs TPF/TPS**

If TP-only is available, overlay TP-only and EP4 PCE.

---

## 27. Suggested Main Table

| R / Method | Quality | TPF | Step ms | TPS | PCE | D+C ms | Expert ms | EP ms/commit | Active Experts | Max/Mean Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R=1 SingleBD-equivalent | ... | ... | ... | ... | 1.00 | ... | ... | ... | ... | ... |
| R=2 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| R=3 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| R=4 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 28. Required Final Report Questions

The final `report.md` must answer:

1. Which exact MBD checkpoint/runtime was used?
2. Is the checkpoint MoE, and what is its expert geometry?
3. Was TRUE physical EP4 verified on GPUs 2,3,6,7?
4. How does TPF scale with R?
5. How does step latency scale?
6. How does actual TPS scale?
7. What is PCE at each R?
8. Does EP cost per committed token improve or worsen?
9. How do active-expert union and tokens/expert change?
10. Does rank-graph density change, or was it already dense?
11. Does max-rank imbalance worsen?
12. What fraction of added MultiBD step latency is explained by D+C+Expert?
13. If TP-only ran, does EP4 convert MultiBD parallelism materially worse?
14. What quality change occurs as R grows?
15. Is the effect robust across held-out requests?
16. Which physical mechanism best explains any conversion loss?
17. Final verdict: GO / CONDITIONAL / NO-GO / GENERIC_MULTIBD_ONLY / RUNTIME_BLOCKED.
18. If GO, what is the smallest next EP-aware running-set experiment?

---

## 29. Accuracy / Benchmark Integrity

Use official benchmark evaluators.

For code:

```text
HumanEval+ / MBPP+ pass@1
```

For math:

```text
GSM8K / MATH500
```

Do not execute untrusted generated code outside a resource-limited sandbox.

Freeze prompts before results.

---

## 30. Evidence Integrity

Preserve:

- exact checkpoint revision;
- Diffulex commit/branch;
- any EP bridge patch;
- config for every R;
- frozen sample IDs;
- per-forward metrics;
- route traces;
- clean timing runs;
- quality outputs;
- GPU topology;
- environment versions.

Run consistency tests for:

- committed-token accounting;
- active-block accounting;
- expert ownership;
- rank-edge counting;
- D+C byte arithmetic;
- tracing parity.

Do not silently discard failed requests.

---

## 31. Required Outputs

Recommended directory:

```text
poc_multibd_moe_ep/
├── scripts/
│   ├── inspect_multibd_runtime.py
│   ├── validate_ep4.py
│   ├── freeze_workload.py
│   ├── run_sweep.py
│   ├── trace_ep.py
│   ├── analyze_conversion.py
│   ├── analyze_expert_workingset.py
│   ├── analyze_rank_pressure.py
│   ├── optional_tp_control.py
│   └── evaluate_quality.py
├── results/
│   ├── environment.json
│   ├── frozen_manifest.json
│   ├── per_forward.csv
│   ├── per_request.csv
│   ├── ep_metrics.csv
│   ├── conversion_summary.json
│   ├── quality.json
│   └── validity.json
├── figures/
│   ├── tpf_tps_step_vs_running_set.png
│   ├── conversion_efficiency.png
│   ├── ep_cost_per_commit.png
│   ├── expert_workingset.png
│   ├── rank_pressure.png
│   ├── latency_breakdown.png
│   └── quality_vs_parallelism.png
├── prior_overlap.md
└── report.md
```

Do not overwrite prior PoCs.

---

## 32. Git Hygiene

Preserve unrelated dirty/staged work.

Suggested branch:

```text
flashvep/multibd-moe-ep-poc
```

Commit/push only this PoC's files.

Final handoff must include:

- verdict;
- checkpoint/runtime;
- TRUE EP4 validation;
- R=1..4 TPF;
- step latency;
- TPS;
- PCE;
- D+C/expert cost per commit;
- expert working-set change;
- rank pressure;
- quality;
- TP-only control status;
- branch;
- commit SHA;
- report path.

---

## 33. Final Criterion

Continue only if:

```text
MultiBD increases algorithmic useful-token parallelism
    ->
physical EP cost grows disproportionately
    ->
TPF gain converts poorly into real TPS
    ->
EP communication/expert/rank behavior explains the conversion loss
    ->
quality remains acceptable
```

If MultiBD scales efficiently under TRUE EP4, close the direction rather than inventing an EP-aware controller.

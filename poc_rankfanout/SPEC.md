# Rank-Fanout-Aware EP Communication PoC Specification

**Project:** MLLM MoE EP
**Direction:** Dynamic `AllGather+ReduceScatter (AGRS)` ↔ sparse `All-to-All (A2A)` communication based on destination-rank fanout
**Primary model:** Qwen3-VL-30B-A3B-Instruct
**Primary regime:** GPU-only MoE Expert Parallel inference, prefill first
**Status:** PoC / hypothesis validation
**Non-negotiable constraint:** CPU↔GPU expert offloading is completely out of scope.

---

## 1. Executive Summary

This PoC tests one specific systems hypothesis:

> **MoE can be sparse in expert space while being effectively dense in EP-rank communication space.**

For a token routed to `top-k = 8` experts under `EP = 4`, the eight experts can easily span all four destination ranks. In that case, a sparse A2A backend may lose much of its communication-volume advantage because the token still needs to reach almost every EP rank, while continuing to pay routing, packing, metadata, synchronization, and dispatch/combine overheads.

The key variable is therefore not `top-k` alone, but the number of **unique destination EP ranks** touched by each token:

\[
F_{t,l} = \left|\left\{\operatorname{rank}(e): e \in E_{t,l}^{top-k}\right\}\right|
\]

where `F` is the **destination-rank fanout**.

The research question is:

> **Does destination-rank fanout causally determine which EP communication backend is faster, and does real Qwen3-VL inference contain enough layer/step-level regime variation that a dynamic AGRS↔A2A selector has meaningful TTFT headroom over the best static backend?**

The PoC must answer this before implementing a real dynamic backend switch.

The validation sequence is intentionally strict:

1. **Runtime-path sanity:** prove that the selected AGRS and DeepEP runs actually execute the intended EP communication paths.
2. **Controlled synthetic experiment:** hold token count, top-k, total expert assignments, rank load, expert load, dtype, hidden size, and topology as constant as possible while varying only destination-rank fanout `F = 1,2,3,4`.
3. **Real Qwen3-VL routing characterization:** measure per-layer/per-step fanout distributions on real text/image workloads.
4. **Static backend benchmarking:** run the same workloads under AGRS and DeepEP HT, and DeepEP LL only if valid and stable.
5. **Perfect dynamic oracle:** estimate the best possible per-invocation backend selection and compare it against the best static backend.
6. **Amdahl/TTFT projection:** explicitly translate communication-level headroom into a request-level TTFT upper bound.

### Primary decision rule

- **GO:** projected best-static → perfect-dynamic TTFT improvement is approximately **12–15% or larger**, with robust backend winner changes explained by fanout.
- **HOLD / investigate:** approximately **5–12%**, or a promising communication-only oracle that does not yet translate cleanly to TTFT.
- **NO-GO:** **≤5% projected TTFT headroom**, one backend wins almost everywhere, real fanout lacks useful variation, or the synthetic causal relationship is absent.

The perfect dynamic oracle is an **optimistic upper bound**, not a claimed implementation result.

---

# 2. Motivation

## 2.1 Standard intuition

In MoE EP, each token selects a small number of experts. Sparse A2A communication is attractive because a token should only be sent to ranks that own its selected experts.

A simplified communication-volume intuition is:

\[
V_{AGRS} \propto EP \times M \times H
\]

\[
V_{A2A} \propto \sum_{t=1}^{M} F_t \times H
\]

where:

- `EP`: number of expert-parallel ranks,
- `M`: number of tokens handled by the source rank / invocation,
- `H`: hidden-state payload size,
- `F_t`: number of unique destination ranks required by token `t`.

A common approximation uses `top-k` for A2A volume, but for communication the more direct quantity is the number of **unique target ranks**, because multiple selected experts can reside on the same rank.

Therefore:

\[
1 \le F_t \le \min(topk, EP)
\]

and a normalized communication-density measure is:

\[
D_t = \frac{F_t}{EP}
\]

When `D_t ≈ 1`, expert routing is sparse but communication is effectively dense across ranks.

---

## 2.2 Qwen3-VL regime of interest

The current MLLM-EP project substrate has used Qwen3-VL MoE with:

- 48 MoE layers,
- 128 routed experts,
- `top-k = 8`,
- EP configurations including EP4 and EP8,
- routed-expert capture through vLLM.

For the primary PoC, use **EP4** if that is the currently stable 4-GPU configuration.

With linear placement under EP4:

- rank 0: experts `0–31`
- rank 1: experts `32–63`
- rank 2: experts `64–95`
- rank 3: experts `96–127`

For `top-k = 8`, a token can touch:

- `F=1`: all 8 experts on one rank,
- `F=2`: 8 experts distributed over two ranks,
- `F=3`: 8 experts distributed over three ranks,
- `F=4`: experts distributed over all four ranks.

Thus `top-k=8` does **not** imply communication sparsity under EP4.

---

## 2.3 Existing signal and why it is not yet enough

The prior project discussion observed an important request-level signal in a Qwen3 / long-prefill regime:

- DeepEP High Throughput versus
- AllGather + ReduceScatter,

with cases where AGRS appeared roughly 10–18% better in actual TTFT.

That signal motivates this PoC, but it must **not** be treated as proof of the fanout hypothesis.

Possible alternative explanations include:

- different runtime paths,
- different packing/layout overhead,
- different batching behavior,
- DeepEP setup or synchronization overhead,
- GPU topology,
- warmup/CUDA-graph differences,
- kernel launch count,
- unequal rank load,
- expert load skew,
- timing noise,
- request preprocessing or scheduler effects.

The purpose of this PoC is to determine whether **destination-rank fanout is a causal explanatory variable** and whether it produces an exploitable dynamic regime.

---

# 3. Core Hypotheses

## H1 — Expert sparsity does not imply rank communication sparsity

Real Qwen3-VL top-k routing under EP4 frequently produces high destination-rank fanout.

Expected evidence:

- meaningful mass at `F=3` and `F=4`,
- layer-to-layer and/or step-to-step variation,
- multimodal workloads may exhibit a different fanout distribution than text-only workloads.

H1 alone is descriptive and is not enough for a systems contribution.

---

## H2 — Backend relative performance depends on destination-rank fanout

At fixed token count and controlled rank/expert load:

- low fanout should favor sparse A2A,
- high fanout should reduce A2A's communication-volume advantage,
- AGRS may become competitive or faster when communication is nearly dense.

The important target is **backend crossover**, not merely a correlation.

Formally, define for invocation `i`:

\[
\Delta_i = T_i^{A2A} - T_i^{AGRS}
\]

Then:

- `Δ < 0`: A2A wins,
- `Δ > 0`: AGRS wins.

H2 expects `Δ` to increase as fanout increases, after controlling for token count and load.

---

## H3 — Real inference contains multiple communication regimes

Even within one model and one request, effective fanout may differ by:

- layer,
- prefill chunk / iteration,
- batch composition,
- modality,
- decode step,
- active sequence count.

A dynamic method is only interesting if backend winners actually vary at a useful granularity.

If one backend wins almost every invocation, a dynamic selector is unnecessary.

---

## H4 — A dynamic oracle beats the best static backend by enough to matter at TTFT

The main systems gate is not communication-kernel speedup alone.

The project has already experienced cases where a large MoE-local improvement produced negligible end-to-end improvement because the optimized region was a small fraction of TTFT.

Therefore the PoC must estimate:

1. communication/MoE-local oracle headroom, and
2. **request-level TTFT headroom after Amdahl adjustment**.

Only the second justifies implementation investment.

---

# 4. Scope

## 4.1 In scope

- GPU-only EP inference.
- Qwen3-VL-30B-A3B-Instruct or the exact currently working Qwen3-VL MoE checkpoint.
- EP4 as the primary configuration.
- Prefill as the primary phase.
- AGRS:
  - vLLM `allgather_reducescatter`.
- Sparse A2A:
  - vLLM `deepep_high_throughput`.
- Optional:
  - `deepep_low_latency` if the installed stack supports it correctly and it is meaningful for the tested phase.
- Synthetic controlled routing.
- Real routed-expert traces.
- Per-layer/per-step fanout analysis.
- Static backend benchmarking.
- Perfect dynamic oracle calculation.
- TTFT / scheduled-to-first-token / MoE communication timing where available.
- Runtime-path verification using profiler/NVTX/kernel evidence.

---

## 4.2 Explicitly out of scope

Do **not** implement or evaluate the following in this PoC:

- CPU↔GPU expert offloading.
- CPU-assisted expert caching.
- GPU expert weight movement or LLEP.
- Expert replication.
- Dynamic expert placement.
- EPLB.
- token merging/pruning.
- route modification.
- top-k modification.
- speculative decoding.
- KV-cache methods.
- TP/EP dynamic reconfiguration.
- generic load-balancing algorithms.
- production-quality dynamic AGRS↔A2A switching.
- new custom communication kernels.
- backend autotuning across many unrelated backends.

This is a **causal validation and oracle PoC**, not the final method.

---

# 5. Definitions and Metrics

## 5.1 Token-level destination-rank fanout

For token `t`, layer `l`:

\[
F_{t,l} =
\left|
\left\{
rank(e) \mid e \in routed\_experts[t,l,:]
\right\}
\right|
\]

For EP4:

\[
F_{t,l} \in \{1,2,3,4\}
\]

---

## 5.2 Fanout density

\[
D_{t,l} = F_{t,l}/EP
\]

For EP4:

- F1 → 0.25
- F2 → 0.50
- F3 → 0.75
- F4 → 1.00

---

## 5.3 Invocation-level fanout summary

For each MoE invocation:

- token count `M`,
- mean fanout,
- median fanout,
- p90 fanout,
- p99 fanout,
- fanout histogram:
  - `frac_f1`,
  - `frac_f2`,
  - `frac_f3`,
  - `frac_f4`,
- full-fanout ratio:
  - `frac_full = P(F=EP)`,
- average fanout density,
- number of active src→dst rank pairs.

The primary scalar should be:

\[
\bar F = \frac{1}{M}\sum_t F_t
\]

but do not discard the full histogram. Two invocations can have the same mean with different tails.

---

## 5.4 Rank load

For rank `r`:

\[
L_r = \sum_{t,k} 1[rank(e_{t,k}) = r]
\]

Log:

- mean rank load,
- max rank load,
- max/mean imbalance,
- coefficient of variation,
- optionally p95 across ranks when EP size is larger.

This is essential because a backend latency change attributed to fanout could actually come from rank imbalance.

---

## 5.5 Expert load

For expert `e`:

\[
L_e = \sum_{t,k} 1[e_{t,k}=e]
\]

Log at least:

- max expert load,
- mean expert load,
- max/mean expert imbalance,
- number of active experts,
- optional HHI or normalized entropy.

Synthetic experiments must keep expert load comparable across fanout conditions.

---

## 5.6 Backend delta

For aligned invocation `i`:

\[
\Delta_i = T_i^{A2A} - T_i^{AGRS}
\]

Also report normalized delta:

\[
\Delta_i^{rel} =
\frac{T_i^{A2A}-T_i^{AGRS}}
{\min(T_i^{A2A},T_i^{AGRS})}
\]

Use raw milliseconds for primary reporting; percentages are secondary.

---

# 6. PoC Architecture

Create this as a **new isolated PoC area** rather than modifying `poc_flashvep`.

Recommended structure:

```text
poc_rankfanout/
├── README.md
├── SPEC.md
├── rankfanout/
│   ├── __init__.py
│   ├── routing.py
│   ├── metrics.py
│   ├── trace_schema.py
│   └── oracle.py
├── scripts/
│   ├── check_runtime_path.py
│   ├── generate_synthetic_routes.py
│   ├── bench_synthetic_backends.py
│   ├── capture_real_routes.py
│   ├── bench_real_static_backends.py
│   ├── analyze_fanout.py
│   └── analyze_oracle.py
├── tests/
│   ├── test_routing.py
│   ├── test_synthetic_balance.py
│   ├── test_metrics.py
│   └── test_oracle.py
├── configs/
│   ├── synthetic_ep4.yaml
│   └── real_qwen3_ep4.yaml
├── results/
└── reports/
```

Exact filenames may be adapted to the existing repository conventions, but preserve separation from old FlashVEP results.

---

# 7. Stage 0 — Runtime Path and Environment Gate

This stage is mandatory.

A previous project lesson is that a configured backend string does not by itself prove that the request actually traverses the intended communication primitive. TP/DP/EP flattening, DPEP versus no-DPEP paths, or model configuration can change the real execution path.

## 7.1 Record environment

Save:

- git commit / branch,
- model path and exact model config,
- vLLM version and commit if locally patched,
- PyTorch version,
- CUDA version,
- NCCL version,
- DeepEP version / commit if available,
- GPU model,
- GPU count,
- `CUDA_VISIBLE_DEVICES`,
- `nvidia-smi topo -m`,
- TP / DP / EP / PP configuration,
- expert placement,
- dtype,
- max model length,
- max batched tokens,
- CUDA graph / eager mode setting,
- all relevant environment variables.

No CPU offload-related setting should be enabled.

---

## 7.2 Primary topology

Prefer the currently stable four-GPU setup.

Primary target:

```text
EP = 4
top-k = 8
experts = 128
linear expert placement
```

If TP/DP are required by the current vLLM path, explicitly document the **effective EP mapping** rather than assuming EP equals the CLI-visible TP size.

The experiment is invalid if the actual expert ownership differs from the fanout calculation.

---

## 7.3 Verify backend availability

Verify the installed stack supports:

- `allgather_reducescatter`,
- `deepep_high_throughput`,
- optionally `deepep_low_latency`.

Do not spend substantial time making LL work if it is incompatible. AGRS versus DeepEP HT is sufficient for the first pass.

---

## 7.4 Verify actual kernels / path

For at least one short run per backend:

### AGRS

Prove that the intended DPEP AGRS path is active using one or more of:

- vLLM runtime class / manager name,
- NVTX ranges,
- Nsight Systems kernel trace,
- NCCL collective names,
- internal debug logs.

Expected evidence should be consistent with actual all-gather / reduce-scatter-style prepare/finalize communication.

### DeepEP HT

Similarly prove DeepEP dispatch/combine kernels are active.

Do not compare performance if one run silently falls back to another path.

---

## 7.5 Stage 0 failure condition

Stop the PoC and report BLOCKED if:

- AGRS and DeepEP are not exercising comparable EP semantics,
- expert ownership mapping is inconsistent,
- one backend silently falls back,
- DeepEP is unstable or unsupported,
- a backend changes model outputs materially,
- runtime path cannot be verified.

Do not proceed with ambiguous backend labels.

---

# 8. Stage A — Controlled Synthetic Fanout Experiment

## 8.1 Objective

Test the causal statement:

> At fixed work and load, changing only the number of destination ranks touched by each token changes the relative performance of AGRS and sparse A2A.

This is the most important mechanism experiment.

---

## 8.2 Keep these variables fixed

Across fanout conditions:

- same GPU topology,
- same EP size,
- same hidden size,
- same dtype,
- same top-k,
- same number of tokens `M`,
- same total expert assignments `M × top-k`,
- same router-weight distribution,
- same total rank load,
- same rank-load balance,
- same expert-load distribution as closely as practical,
- same number of active experts if possible,
- same MoE GEMM shape distribution if practical,
- same warmup,
- same process lifetime,
- same synchronization policy,
- same backend-specific configuration except the backend itself.

The experiment is not valid if `F=4` simply creates more expert work or worse rank imbalance than `F=1`.

---

## 8.3 Construct controlled routing for EP4 / top-k 8

Assume 32 experts per rank under linear placement.

Per-token patterns:

### Fanout 1

```text
8 selected experts from one destination rank.
```

Example token:

```text
rank 0: 8 experts
rank 1: 0
rank 2: 0
rank 3: 0
```

Across the full token set, rotate the home rank so each rank receives the same total number of expert assignments.

---

### Fanout 2

Prefer:

```text
4 + 4 assignments across two destination ranks.
```

Rotate rank pairs uniformly:

```text
(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
```

or use a deterministic balanced schedule.

---

### Fanout 3

Prefer:

```text
3 + 3 + 2 assignments across three ranks.
```

Rotate the rank receiving 2 assignments and rotate the omitted rank.

---

### Fanout 4

Prefer:

```text
2 + 2 + 2 + 2 assignments across all four ranks.
```

---

## 8.4 Expert-level balancing

Within each rank, rotate selected expert IDs so that no small subset of experts becomes hot only because of the fanout construction.

Generate and assert:

- equal total assignments per rank within tolerance,
- approximately equal assignments per expert,
- exactly 8 distinct expert IDs per token,
- exact requested fanout for every token,
- no invalid expert IDs.

Synthetic generator tests must fail loudly if these constraints are violated.

---

## 8.5 Token-count sweep

Use a geometric sweep broad enough to cover small through long-prefill-like regimes.

Recommended first pass:

```text
M ∈ {128, 256, 512, 1024, 2048, 4096, 8192}
```

If memory or runtime makes 8192 impractical, retain at least:

```text
256, 1024, 4096
```

Because the prior signal came from a long-prefill regime, include a token scale near the previous 8K case whenever possible.

---

## 8.6 Backends

Required:

```text
AGRS
DeepEP High Throughput
```

Optional:

```text
DeepEP Low Latency
```

Only include LL if:

- runtime path is verified,
- it works under the target topology,
- comparison semantics are fair.

Do not delay the PoC for LL.

---

## 8.7 Measurement boundaries

Measure separately where possible:

1. routing/packing metadata,
2. dispatch / prepare,
3. local expert compute,
4. combine / finalize,
5. total MoE boundary.

The core backend-sensitive metric should include the complete communication-related path, not only the visible network kernel.

A backend with a faster collective but expensive packing should not be incorrectly labeled faster.

---

## 8.8 Repetition methodology

For each `(M, fanout, backend)` condition:

- warm up sufficiently,
- use at least 30 measured repetitions for a quick pass,
- prefer 100 repetitions for final synthetic plots,
- synchronize only where required for correct timing,
- avoid adding a global sync that changes the production path,
- randomize or alternate backend experiment ordering to reduce drift,
- repeat the full matrix at least 3 independent times if the signal is small.

Report:

- median,
- p90,
- p99,
- IQR or confidence interval,
- run-to-run variance.

---

## 8.9 Synthetic outputs

Produce:

### Plot A1 — latency vs fanout

For each `M`:

```text
x = fanout 1..4
y = communication or full MoE latency
series = AGRS, DeepEP HT [, DeepEP LL]
```

### Plot A2 — backend delta vs fanout

```text
x = mean fanout
y = DeepEP - AGRS latency
```

A zero crossing is the desired crossover signal.

### Plot A3 — winner map

```text
rows = M
columns = fanout
cell = fastest backend / speedup
```

### Table A1

Include:

- M,
- requested fanout,
- measured mean fanout,
- rank imbalance,
- expert imbalance,
- AGRS p50,
- DeepEP HT p50,
- optional LL p50,
- winner,
- relative gap.

---

## 8.10 Synthetic success criterion

Strong evidence for H2 requires:

1. load-control checks pass,
2. backend delta changes systematically with fanout,
3. preferably a backend winner crossover,
4. effect reproduces across token counts or has an interpretable `M × fanout` boundary.

### NO-GO after Stage A

Strongly consider stopping if:

- latency is essentially insensitive to fanout,
- one backend dominates across all fanout values and token scales,
- apparent fanout effect disappears after load balancing,
- observed effect is smaller than normal run-to-run noise.

---

# 9. Stage B — Real Qwen3-VL Fanout Characterization

## 9.1 Objective

Determine whether the synthetic mechanism appears in real routing and whether fanout varies enough to justify dynamic selection.

---

## 9.2 Capture mechanism

Use the existing vLLM routed-expert capture path if still available:

```text
enable_return_routed_experts=True
routed_experts shape ≈ [seq, layer, topk]
```

Do not assume the exact API is unchanged; inspect current repository and installed vLLM.

Map each expert ID to its actual EP rank using the runtime expert map.

For simple EP4 linear placement:

```python
rank = expert_id // 32
```

but do not hard-code this unless runtime verification confirms it.

---

## 9.3 Workload groups

Prioritize the exact workload where AGRS previously showed the 10–18% TTFT signal if those prompts/configs can be recovered.

At minimum include:

### W1 — text-only prefill

- moderate length,
- long length if available.

### W2 — single-image MLLM prefill

- real images,
- same preprocessing resolution/config across runs.

### W3 — multi-image or vision-heavy prefill

Only if already available in the project and easy to reproduce.

### W4 — long-prefill target

Aim for total prefill scale near the prior ~8K regime.

Decode is secondary. If cheap to collect, record it separately rather than mixing it with prefill.

---

## 9.4 Per-invocation trace schema

Write JSONL or Parquet/CSV with one row per MoE invocation.

Minimum fields:

```text
run_id
backend
request_id
batch_id
phase                 # prefill / decode
step_id
layer_id
num_tokens
top_k
ep_size
num_experts
expert_placement_hash
mean_fanout
median_fanout
p90_fanout
p99_fanout
frac_f1
frac_f2
frac_f3
frac_f4
frac_full_fanout
active_src_dst_pairs
rank_load_mean
rank_load_max
rank_load_max_over_mean
rank_load_cv
expert_load_mean
expert_load_max
expert_load_max_over_mean
active_experts
dispatch_ms
expert_ms
combine_ms
moe_total_ms
timestamp
```

Add backend-specific metadata when useful.

Do not fabricate unavailable timings; use null and document the missing boundary.

---

## 9.5 Request-level schema

For each request/batch:

```text
request_id
workload_group
num_text_tokens
num_vision_tokens
total_prefill_tokens
num_output_tokens
backend
ttft_ms
scheduled_to_first_token_ms   # if available
e2e_ms
mean_layer_fanout
p90_layer_fanout
frac_layers_high_fanout
```

Separate preprocessing time from model TTFT if possible.

---

## 9.6 Real-routing outputs

### Plot B1 — layer fanout profile

```text
x = layer 0..47
y = mean destination-rank fanout
```

Use separate traces for text-only and vision-heavy workloads.

### Plot B2 — fanout heatmap

```text
rows = request / step
columns = layer
value = mean fanout
```

### Plot B3 — histogram

Distribution of token-level and invocation-level fanout.

### Plot B4 — fanout vs rank imbalance

This verifies that fanout is not merely a proxy for load skew.

---

## 9.7 Stage B interpretation

Dynamic backend selection needs **regime diversity**.

Promising pattern:

```text
some invocations/layers: F ≈ 1.5–2.5
others:                F ≈ 3.5–4.0
```

Weak pattern:

```text
almost every layer/request: F ≈ 4
```

If nearly everything is full fanout, that may explain why AGRS can beat A2A, but it argues for a **better static backend**, not a dynamic method.

Likewise, if almost everything is low fanout, static A2A is likely enough.

---

# 10. Stage C — Static Backend Benchmark on Real Workloads

## 10.1 Fairness rules

Run the exact same prompt set under each backend.

Keep fixed:

- model,
- weights,
- dtype,
- seed where meaningful,
- GPU set,
- TP/DP/EP,
- batch construction,
- max batched tokens,
- scheduler settings,
- expert placement,
- output length,
- image preprocessing,
- CUDA graph/eager setting,
- warmup,
- request ordering.

Do not enable EPLB.

---

## 10.2 Primary metrics

### Request-level

- actual TTFT,
- scheduled-to-first-token if already instrumented,
- p50/p90/p99 TTFT,
- throughput as a secondary metric.

### MoE-level

- dispatch / prepare,
- combine / finalize,
- communication boundary,
- full MoE boundary.

The project should not claim success from communication-only speedup if TTFT is flat.

---

## 10.3 Paired comparisons

Prefer paired comparisons per request/workload.

For request `r`:

\[
\Delta TTFT_r = TTFT_r^{A2A} - TTFT_r^{AGRS}
\]

Relate the delta to:

- average fanout,
- high-fanout layer fraction,
- token count,
- rank imbalance.

A useful plot is:

```text
x = request/invocation mean fanout
y = DeepEP TTFT or comm latency - AGRS equivalent
```

Stratify by token-count bucket.

---

# 11. Stage D — Perfect Dynamic Oracle

## 11.1 Why oracle first

vLLM currently exposes backend selection primarily as an engine/server configuration. A real per-layer/per-step switch can require changes to:

- workspace allocation,
- communication manager selection,
- input/output layout,
- CUDA graph assumptions,
- synchronization,
- dispatch/combine metadata.

Implementing this before measuring the upper bound is too expensive.

Therefore the PoC computes a **perfect dynamic oracle** first.

---

## 11.2 Invocation alignment

For each backend, collect comparable invocation records and align using a stable key such as:

```text
(workload_id, request_id, phase, step_id, layer_id, num_tokens)
```

Also verify routing equivalence.

If scheduler behavior makes direct alignment unreliable, use a route/shape replay benchmark rather than pretending the records are aligned.

Discard or separately report unmatched invocations.

---

## 11.3 Best static backend

For backend set `B` and invocations `i`:

\[
T_{static}(b) = \sum_i T_i^b
\]

\[
T_{best-static} = \min_{b \in B} T_{static}(b)
\]

---

## 11.4 Zero-cost perfect dynamic oracle

\[
T_{oracle,comm} =
\sum_i \min_{b \in B} T_i^b
\]

This assumes:

- perfect knowledge,
- zero selection overhead,
- zero backend-switch overhead,
- no cross-invocation interference,
- compatible layouts.

It is intentionally optimistic.

Report:

\[
Improvement_{oracle} =
1 - \frac{T_{oracle}}{T_{best-static}}
\]

and speedup:

\[
Speedup_{oracle} =
\frac{T_{best-static}}{T_{oracle}}
\]

Do not confuse the two.

---

## 11.5 Oracle granularity

Compute at least:

### O1 — per-layer/per-invocation oracle

Choose backend independently for every MoE invocation.

This is the maximum headroom.

### O2 — per-step oracle

One backend choice for all MoE layers in the same prefill/decode step.

This is more implementable.

### O3 — per-request oracle

One backend for the whole request.

This tests whether a much simpler request-level selector is already sufficient.

If O3 captures nearly all of O1, per-layer switching is likely unnecessary.

This comparison is important for novelty and implementation complexity.

---

# 12. Amdahl-Adjusted TTFT Oracle

## 12.1 Required because local speedup can be misleading

Suppose backend switching only changes MoE communication.

Let a request under the best static backend have:

\[
TTFT_{static} =
T_{unaffected} + T_{comm,static}
\]

Then an optimistic dynamic request estimate is:

\[
TTFT_{oracle} =
T_{unaffected} + T_{comm,oracle}
\]

or equivalently:

\[
TTFT_{oracle} =
TTFT_{static}
- T_{comm,static}
+ T_{comm,oracle}
\]

If full MoE timing rather than pure communication timing is the trustworthy aligned region, substitute that boundary consistently.

---

## 12.2 Report both local and projected gains

For every workload report:

```text
communication-only oracle improvement
MoE-boundary oracle improvement
projected TTFT oracle improvement
```

The **projected TTFT improvement** is the primary gate for further implementation.

Example interpretation:

```text
comm oracle       = 25%
MoE oracle        = 18%
projected TTFT    = 3%
=> NO-GO for dynamic backend implementation
```

This avoids repeating prior Amdahl failures.

---

# 13. Fanout-Based Explainability Analysis

The PoC should not end at "oracle says backend changes."

We need to show that **fanout explains the change**.

## 13.1 Basic analysis

For aligned invocations:

- bin by mean fanout,
- within token-count buckets,
- compare median backend delta.

Example bins for EP4:

```text
[1.0, 1.5)
[1.5, 2.5)
[2.5, 3.5)
[3.5, 4.0]
```

Adapt bins to actual data.

---

## 13.2 Control for confounders

At minimum inspect backend delta versus:

- fanout,
- M,
- rank max/mean,
- expert max/mean,
- phase.

A simple regression is acceptable as analysis, not as the final method:

\[
\Delta =
\beta_0
+\beta_1 \bar F
+\beta_2 M
+\beta_3 rank\_imbalance
+\beta_4 expert\_imbalance
+\epsilon
\]

Do not oversell statistical significance from a small sample.

The desired qualitative result is that `β1` is stable and the winner boundary is interpretable.

---

## 13.3 Simple fanout threshold oracle

As a bridge toward a future method, evaluate an offline threshold:

```text
if mean_fanout >= τ:
    AGRS
else:
    DeepEP_HT
```

Search `τ` only on a training/calibration split and evaluate on held-out traces.

This is **analysis only**. Do not integrate the dynamic selector into vLLM during this PoC unless the oracle gate already passes.

Compare:

```text
best static
perfect oracle
fanout-threshold selector
```

If a one-dimensional fanout threshold recovers most oracle headroom, the story is strong.

If it does not, fanout may be descriptive but insufficient for backend selection.

---

# 14. Decision Gates

## Gate G0 — Runtime validity

PASS only if both backends execute the intended EP communication path.

Otherwise: **BLOCKED**.

---

## Gate G1 — Synthetic causal signal

PASS if controlled fanout materially changes backend relative latency.

Best outcome:

- DeepEP wins at low fanout,
- AGRS wins at high fanout,
- crossover moves with M in an interpretable way.

If no causal fanout effect: **NO-GO**.

---

## Gate G2 — Real fanout diversity

PASS if real Qwen3-VL workloads exhibit meaningful layer/step/request variation.

Warning signs:

- >90–95% of invocations are effectively full fanout,
- or >90–95% stay in one narrow fanout regime.

Such results may justify a static backend choice but not a dynamic method.

---

## Gate G3 — Real backend winner diversity

PASS if both AGRS and A2A win meaningful subsets of aligned real invocations.

A practical initial requirement:

```text
minority winner share >= ~10%
```

Prefer a stronger split.

If one backend wins >95% of invocations, dynamic selection is probably unnecessary.

---

## Gate G4 — Oracle headroom

### GO

Continue to actual dynamic-switch implementation only if:

- projected TTFT oracle improvement is roughly **12–15%+**,
- the result is robust across repeated runs,
- fanout explains a meaningful part of the backend crossover,
- a practical selector appears capable of recovering a substantial fraction of the oracle.

### HOLD

- projected TTFT: **5–12%**, or
- local communication oracle is strong but request-level decomposition is still uncertain.

Only perform low-cost follow-up measurements.

### NO-GO

- projected TTFT headroom **≤5%**,
- or gains are within noise,
- or no real winner diversity,
- or fanout does not explain backend behavior.

Do not build a production dynamic backend switch after a NO-GO.

---

# 15. Experimental Matrix

## 15.1 Synthetic matrix

Primary:

```text
EP = 4
top-k = 8
F = {1,2,3,4}
M = {128,256,512,1024,2048,4096,8192}
backend = {AGRS, DeepEP_HT}
```

Optional:

```text
backend += DeepEP_LL
```

Total required primary cells:

```text
4 × 7 × 2 = 56
```

Run warmup + repeated measurements per cell.

---

## 15.2 Real matrix

Start small:

```text
workload:
  text-only
  single-image
  long/8K-like target
backend:
  AGRS
  DeepEP_HT
repetitions:
  >= 3 full paired runs
```

Only expand to multi-image or decode after the first analysis shows useful headroom.

---

# 16. Measurement Pitfalls and Required Safeguards

## 16.1 Backend name is not runtime proof

Always verify actual kernels/path.

---

## 16.2 EP mapping must be runtime-derived

Do not assume:

```text
expert_id // experts_per_rank
```

if custom placement, EP flattening, TP/DP interaction, or a patched expert map is active.

Hash and log the expert-to-rank map.

---

## 16.3 Same top-k does not mean same network payload

Multiple experts on one rank may allow token deduplication in some A2A implementations and may not in others.

This is part of the phenomenon, but the implementation semantics must be documented.

---

## 16.4 Keep rank load separate from fanout

High fanout may also produce better or worse load balance.

Synthetic routing must control load, and real analysis must log both variables.

---

## 16.5 Keep expert GEMM work constant

Every synthetic token should still have exactly 8 expert assignments.

Do not accidentally turn a fanout experiment into a compute-volume experiment.

---

## 16.6 Avoid scheduler confounding

For backend-to-backend request comparisons:

- fixed prompts,
- fixed batching,
- fixed output count,
- controlled arrival pattern,
- preferably offline deterministic runs first.

---

## 16.7 Prefill and decode are different regimes

Do not mix them into one latency distribution.

DeepEP HT and LL target different regimes.

Primary claim should begin with prefill if that is where the prior TTFT signal exists.

---

## 16.8 Warmup and CUDA graph effects

Log:

- eager versus graph mode,
- first-iteration behavior,
- graph capture/replay boundaries.

Exclude initialization from steady-state results unless TTFT deployment semantics require including it.

---

## 16.9 Topology matters

Record physical GPU topology.

A fanout result on PCIe-only topology is not automatically transferable to NVLink.

---

## 16.10 Instrumentation overhead

Measure an uninstrumented baseline and estimate instrumentation overhead.

If fine-grained timing changes TTFT by several percent, use profiling runs for attribution and clean runs for final request-level latency.

---

# 17. Validation Tests

Before expensive GPU runs, implement CPU/unit tests for fanout logic.

## 17.1 Fanout tests

Given routed experts and mapping:

- correct fanout 1,
- correct fanout 2,
- correct fanout 3,
- correct fanout 4,
- duplicate experts/ranks handled correctly.

---

## 17.2 Synthetic generator tests

For each F:

- every token has exactly `top-k=8` distinct experts,
- every token has exactly requested destination fanout,
- total assignments fixed,
- rank load within strict tolerance,
- expert load within declared tolerance.

---

## 17.3 Oracle tests

Use toy latency matrices where expected:

- best static backend is known,
- per-invocation oracle is known,
- speedup and improvement formulas are checked,
- missing/unmatched invocation behavior is tested.

---

## 17.4 Trace alignment tests

Detect:

- duplicate keys,
- missing layer records,
- routing mismatch across backend runs,
- token-count mismatch,
- different expert placement hashes.

Oracle must refuse invalid alignments rather than silently averaging them.

---

# 18. Expected Deliverables

The PoC should produce the following.

## 18.1 Code

- synthetic routing generator,
- fanout metrics,
- route capture,
- static benchmark runners,
- oracle analyzer,
- unit tests.

---

## 18.2 Raw/derived data

- environment snapshot,
- runtime-path evidence,
- synthetic JSONL/CSV,
- real fanout traces,
- backend timing traces,
- oracle summary JSON.

---

## 18.3 Figures

Required:

1. **Synthetic latency vs fanout**
2. **Synthetic backend delta vs fanout**
3. **M × fanout winner heatmap**
4. **Real Qwen layer-wise fanout**
5. **Real fanout heatmap / distribution**
6. **Backend delta vs real fanout**
7. **Best-static vs perfect oracle vs fanout-threshold selector**
8. **Communication oracle vs projected TTFT oracle**

---

## 18.4 Final report

Create:

```text
poc_rankfanout/reports/final_report.md
```

with:

- exact environment,
- hypothesis,
- path verification,
- synthetic results,
- real fanout results,
- static backend results,
- oracle,
- Amdahl projection,
- confounders,
- GO/HOLD/NO-GO decision.

The report must clearly distinguish:

```text
measured
estimated
oracle upper bound
speculative future method
```

---

# 19. Recommended Execution Order

Do not jump directly to full-model dynamic implementation.

Use this order:

### Step 1
Inspect current repository HEAD and current vLLM/DeepEP stack.

### Step 2
Create `poc_rankfanout/` and environment snapshot.

### Step 3
Verify AGRS and DeepEP HT runtime paths.

### Step 4
Implement/test fanout metrics and synthetic route generator.

### Step 5
Run a tiny synthetic smoke:

```text
M = 256
F = 1,4
backend = AGRS, DeepEP_HT
```

If there is no measurable difference, diagnose before expanding.

### Step 6
Run full synthetic fanout × M matrix.

### Step 7
Analyze H2 and decide whether real-model work is justified.

### Step 8
Capture real Qwen3-VL fanout traces.

### Step 9
Benchmark real workloads under static AGRS and DeepEP HT.

### Step 10
Align invocations and compute O1/O2/O3 perfect oracles.

### Step 11
Compute Amdahl-adjusted TTFT oracle.

### Step 12
Fit/evaluate a simple held-out fanout threshold selector offline.

### Step 13
Issue GO/HOLD/NO-GO.

Only after **GO** should a new phase design actual per-layer/per-step communication backend switching.

---

# 20. What a Strong Positive Result Looks Like

An ideal result would resemble:

```text
Synthetic:
M=4096
F≈1–2: DeepEP HT clearly faster
F≈4:   AGRS clearly faster

Real Qwen3:
some layers/steps cluster near F≈2
others near F≈4

Static:
AGRS wins some workloads
DeepEP wins others

Invocation-level:
backend winner flips systematically with fanout

Oracle:
best-static -> perfect per-layer oracle:
>= 12–15% projected TTFT improvement

Simple threshold:
recovers most of oracle headroom on held-out traces
```

This supports the narrative:

> **Expert sparsity is not communication sparsity. Destination-rank fanout determines the effective EP communication regime, but current serving systems largely commit to a static communication backend.**

---

# 21. What a Negative Result Looks Like

Examples:

### Case A — Full fanout everywhere

```text
F ≈ 4 for almost every token/layer/request
AGRS wins almost always
```

Interpretation:

- useful explanation for AGRS performance,
- but dynamic backend selection is unnecessary,
- choose/static-optimize AGRS instead.

### Case B — A2A wins everywhere

```text
fanout changes but DeepEP HT remains faster
```

Interpretation:

- fanout is not enough to overcome A2A implementation efficiency,
- no dynamic method.

### Case C — Communication oracle large, TTFT oracle small

```text
comm oracle: 20%
TTFT projection: 3%
```

Interpretation:

- Amdahl-limited,
- NO-GO for E2E latency research direction.

### Case D — Winner changes but fanout does not explain it

Interpretation:

- there may be another useful predictor such as token count, rank imbalance, layout, or phase,
- but the specific **rank-fanout-aware** hypothesis is not validated.

Do not rename the effect after the fact without a new hypothesis and controlled experiment.

---

# 22. Future Phase Only If PoC = GO

If all gates pass, the next phase can study an actual dynamic communication selector.

Possible selector inputs:

```text
M
mean fanout
frac_full_fanout
rank load imbalance
phase
```

A minimal policy could be:

```text
if phase == prefill and mean_fanout >= tau(M):
    AGRS
else:
    DeepEP
```

Future implementation questions:

- can routing-derived fanout be known early enough without duplicated work?
- can backend workspaces be preallocated?
- can both managers coexist?
- what is backend-switch overhead?
- can CUDA graphs tolerate switching?
- what granularity is best:
  - request,
  - step,
  - layer?
- does per-step switching recover most per-layer oracle?
- can a threshold be calibrated once per hardware/model pair?

These are deliberately **not** required in the current PoC.

---

# 23. Prior-Art / Novelty Questions for the Next Phase

Before claiming novelty, separately verify:

1. whether existing systems choose communication strategy only at startup/configuration time,
2. whether any system selects the primitive using actual routed destination-rank fanout,
3. whether selection occurs per request, step, or MoE layer,
4. whether communication volume is modeled by unique destination ranks rather than top-k,
5. whether a system already provides an equivalent dynamic strategy factory at MoE-invocation granularity.

The PoC can proceed before this survey is complete, but a paper direction cannot.

---

# 24. Reference Notes

## Current MLLM-EP repository substrate

The repository snapshot already contains evidence that:

- Qwen3-VL MoE uses 48 layers, 128 experts, top-k 8,
- routed experts can be returned as `[seq, layer, topk]`,
- vLLM EP experiments have used `allgather_reducescatter`,
- prior profiling showed that configured backend names must be checked against the actual runtime communication path.

The implementation agent should inspect the **current HEAD** rather than assuming an older snapshot is authoritative.

## vLLM EP backend interface

Current vLLM documentation exposes communication backends including:

- `allgather_reducescatter`,
- `deepep_high_throughput`,
- `deepep_low_latency`,
- additional FlashInfer/MoRI/etc. options.

Reference:
https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/

## NVIDIA communication-volume motivation

NVIDIA's TensorRT-LLM communication discussion distinguishes:

- AllGather + ReduceScatter, whose logical data movement scales with EP size,
- sparse AlltoAll, whose useful destination traffic depends on selected target ranks.

It also explicitly discusses deduplicating a token when multiple selected experts reside on the same destination rank, reinforcing that unique destination-rank count is the relevant communication quantity.

Reference:
https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.html

---

# 25. Final PoC Decision Statement Template

At completion, end the report with exactly one of:

## GO

> Destination-rank fanout causally changes AGRS vs A2A relative performance, real Qwen3-VL inference exhibits meaningful regime variation, and the Amdahl-adjusted perfect dynamic oracle provides ≥12–15% request-level TTFT headroom over the best static backend. Proceed to dynamic selector implementation.

## HOLD

> The fanout hypothesis is partially supported, but projected TTFT headroom is 5–12% or measurement uncertainty remains material. Perform only the listed low-cost follow-up measurements before implementation.

## NO-GO

> The rank-fanout-aware dynamic communication direction does not provide sufficient E2E headroom or backend regime diversity. Do not implement per-layer/per-step AGRS↔A2A switching.

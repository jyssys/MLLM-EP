# Looped Transformers × Multi-GPU: Recurrent KV Placement and Execution — Discovery PoC

**Status:** Experimental contract / characterization first (NOT an implemented method)

**Intended repository:** `/home/esjung/MLLM-EP-github`

**Place this document at:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_multigpu_kv_placement_poc_spec.md`

**Allowed physical NVIDIA GPUs:** **4, 5, 6, 7 only**; on the previously established server these are H100 80 GB / NVLink fabric. Confirm before executing.

**Mandatory model A:** pretrained **`ByteDance/Ouro-2.6B-Thinking`** (or `ByteDance/Ouro-2.6B` only if a documented task/runtime issue prevents Thinking). **Mandatory model B feasibility investigation:** pretrained **`tomg-group-umd/huginn-0125`**, approximately 3.5B, with a structurally different prelude/recurrent-core/coda layout. Obtain and pin exact revisions.

**Candidate fallback for model B:** only if Huginn cannot support meaningful physical multi-GPU execution without engine rewrites, investigate a distinct pretrained, correctly looped architecture/checkpoint from `ifm-ai/xllm-loop` or another verified research release; validate actual architecture, scale, license, checkpoint, and TP feasibility before adoption. `Ouro-1.4B` is acceptable as a *same-family sanity control*, **not** as the second independent model-family result. Do not fabricate a two-family generality claim.

**Serving regime:** batch-1 latency first; concurrent-request capacity and throughput second. Preserve model behavior, pretrained weights, native attention, native KV correctness, and precision. NO new loop training, no KV compression, no learned scheduler, no fresh architecture implementation until the discovery gates pass.

---

## 0. Executive Research Question

**Does weight/head-oriented Tensor Parallelism (TP) match the *actual* KV-memory, compute, and access pattern of different pretrained looped Transformers? If not, can independently placing KV state and recurrent compute across four GPUs yield sufficiently large *physical*, quality-preserving memory–latency–throughput headroom to justify a new inference framework?**

Do **not** assume a new distributed memory architecture is necessary. Existing TP may already be excellent. The PoC must be capable of falsifying the entire direction cheaply.

In particular:

- Parameter sharing across recurrence **does not imply** K/V equality across recurrent passes.
- A 4-way head shard provides approximately one-fourth of the *logical K/V head payload per rank* only when K/V-head layout, TP support, actual cache policy, and divisibility permit. TP also shards projection/MLP weights and changes collective costs. Total KV across four ranks is **not reduced** merely by TP.
- The phrase “48 GiB total KV -> 12 GiB/GPU” is a **hypothetical** proportional split, not an audited measurement of either target model. Derive the correct memory formula from each pinned checkpoint and **count physical allocated cache slots**, not only advertised layers.
- Many looped models have **different** architectures: whole-stack recurrent Ouro vs prelude/recurrent-core/coda Huginn; some have bounded/modulo/frozen/last-available cache policies. There is no universal per-loop KV tensor layout.
- If existing TP2/TP4 already delivers good latency–capacity tradeoffs, report **NO-GO**. Do not invent a loop-aware sharding method to rescue the hypothesis.

### Research axes: keep distinct

1. **Storage capacity:** physical HBM per rank, model weights, loop-indexed K/V, allocator / graph / workspace.
2. **KV access bandwidth and locality:** actual memory reads/writes and effective attention cost, not inferred from logical KV size.
3. **Inter-GPU communication:** real TP collectives and any prospective remote-KV transfer/merge cost.
4. **Critical-path latency:** actual prefill and one-token cached decode TPOT; E2E request wall with the same output budget.
5. **Batch throughput / admission:** feasible concurrent requests at long context; aggregates are distinct from batch-1 latency.
6. **Model fidelity:** cache policy, recurrent step count, precision, autoregressive outputs, no hidden approximations.

---

## 1. Prior Knowledge and Scientific Novelty Boundary

Read and record primary sources with exact code/version dates. Mandatory references:

- **Ouro / Scaling Latent Reasoning via Looped Language Models:** https://ouro-llm.github.io/ and https://huggingface.co/ByteDance/Ouro-2.6B-Thinking ; native Ouro code/vLLM integration.
- **Huginn / Scaling up Test-Time Compute with Latent Reasoning:** https://huggingface.co/tomg-group-umd/huginn-0125 and https://github.com/seal-rg/recurrent-pretraining . Huginn explicitly requires `HuginnDynamicCache` and documents policies such as `compress-s16` or `latest-m4`.
- **FlashLoop:** https://arxiv.org/abs/2609.29812 and official public implementation. Already exploits cross-loop temporal redundancy through sparse updates, attention sparsity, and quantized KV residuals. **Do not claim loop similarity or base-plus-residual compression as novelty.**
- **MELT:** https://arxiv.org/abs/2605.07721 ; new trained architecture with gated shared KV (not training-free).
- **Looped Latent Attention / Hybrid Latent Attention:** learned KV compression; separate from our placement question.
- **Enabling Dynamic Computation in Looped LMs:** best-available KV, model adaptive exit and cache reuse; do not reinvent loop cache fallback.
- **Continuous Depth Batching:** depth-aware active work and batching (single-GPU performance should not be repackaged as multi-GPU novelty).
- **WaveFront / LoopSpec / Parallel Loop Transformer:** recurrence-depth speculation/wavefront; do not claim unqualified inter-loop parallel execution.
- General TP/GQA, standard context parallelism, distributed softmax / Ring Attention, paged KV, vLLM native cache management.
- **Training-Free Looped Transformers (2026):** a *different* topic—converting frozen standard models into loops; not the same as serving pretrained recurrent checkpoints.

Create `prior_overlap.md` with **problem, model, actual hardware, training requirement, method, claims, and gap**. Lack of a dedicated paper on our exact two-model setup is **not** sufficient novelty. The contribution would need a physical multi-GPU problem that neither native TP, existing cache policies, nor conventional KV/CP placement already solves.

---

## 2. Two-Model Feasibility Is a First-Class Deliverable

### Model A: Ouro-2.6B-Thinking

- Official pretrained HF ID: `ByteDance/Ouro-2.6B-Thinking`.
- Public documentation describes shared-stack recurrence, default `total_ut_steps=4`, BF16 recommended, and native vLLM support in specific older versions.
- **Version pin caveat:** Ouro official project states vLLM **v0.26.0 was the last version with native Ouro support**; verify installed versions and isolate new environments. Do not install latest blindly or disturb existing LLaDA2/dInfer environments.
- The model card mentions a smaller *physical/shared-stack* layer count, while released configuration may expose `num_hidden_layers=48` / repeated `layer_types`. **Audit actual module identities, cache slots, recurrent invocations, and checkpoint metadata.** Do not blindly treat every config layer as physically distinct, or multiply 48 by 4 without proof.
- Native vLLM may execute fixed full depth even if Transformers offers adaptive steps. Do not compare adaptive-early-exit and fixed-loop regimes as if identical.
- Prefer fixed recurrent depth 4 for the primary same-quality TP baseline. Additional depth sweeps only if semantically supported and clearly separate.

### Model B: Huginn-0125

- Official pretrained HF ID: `tomg-group-umd/huginn-0125`.
- ~3.5B, prelude + 4-layer recurrent core + coda, with test-time `num_steps` (official examples use 16/32/64).
- Its official **config reports `n_heads=55`, `num_key_value_heads=55`, `head_dim=96`, `n_layers_in_prelude=2`, `n_layers_in_recurrent_block=4`, `n_layers_in_coda=2`.** Standard attention-head TP2/TP4 partition is **not trivially divisible**. The actual vLLM plugin must be audited for padding/partition or special shard semantics. Do NOT assume TP2/TP4 works because the repository advertises vLLM inference.
- Official project has a vLLM V1 plugin (`recurrent-pretraining/vllm`). Inspect real TP support and cache mapping instead of trusting generic HF website auto-generated commands.
- Huginn requires `HuginnDynamicCache`, not a generic cache. Its `compress-s16` cache-sharing strategy is *not* an extra training requirement but may change the model's cache semantics and output—record as **separate secondary baseline**; do not mingle with faithful default full-cache runs.
- For H16/32/64, isolate fixed-depth and adaptive policies. Unrolled compute and KV capacity may differ sharply. Determine effective cache slots from actual runtime.
- If TP2/TP4 would require a new custom model executor or nontrivial 55-head remapping, mark **`MODEL_B_TP_BLOCKED`**; keep an official single-GPU/official plug-in baseline and architectural/memory findings only. Never present a fake physical TP4 result. Consider an alternative **independent** pretrained looped checkpoint with compatible head geometry, if one exists and can be executed with bounded integration.

### Mandatory model comparison table

| Field | Ouro | Huginn |
|---|---|---|
| checkpoint revision & tokenizer | actual | actual |
| physical stored layers vs logical executed layers | audit | audit |
| recurrence topology and default depth | audit | audit |
| Q/KV heads, head dim, GQA/MHA | audit | audit |
| exact TP1/2/4 support | verify | verify, 55-head warning |
| KV cache policy and index tuple | audit | HuginnDynamicCache audit |
| exact KV slot count, bytes/token | measured | measured |
| official serving stack/version | pin | pin |
| output parity with reference | verify | verify |

**Gate 0:** At minimum Ouro native TP1 and TP4 should work and Huginn should at least run as a **real pretrained second model**. If neither model supports adequate cache instrumentation, stop `RUNTIME_BLOCKED`. If Huginn TP is unsupported, this remains an informative **one-family physical multi-GPU** study with model-B caveat, *not* a two-family physical study.

---

## 3. Safety, GPUs, Environment Isolation

Only operate on **physical GPU 4,5,6,7**:

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
nvidia-smi
nvidia-smi topo -m
```

- Record rank->physical GPU->UUID; fabric (NVLink/NVSwitch/PCIe), HBM free and reserved.
- Inspect current GPU owners, including any **user-owned burner jobs** already running. Do not kill, pause, or replace processes without explicit verified permission; never touch other users' processes. If expected GPU state cannot be safely restored, return `RESOURCE_BLOCKED`.
- Prefer new isolated Python/Conda/container environments, *not* mutation of existing dInfer/SGLang environments, model caches, Git staged work, or base packages.
- Distinct engine versions for Ouro and Huginn are acceptable. **Do not make cross-engine absolute speed claims.** All physical TP comparisons for one model must share the same pinned runtime/backend whenever feasible.
- No random weights, synthetic fabricated timing, CPU-only proxy as a physical GPU result, or silent quantization changes.

---

## 4. Cache Geometry and Memory Accounting — Must Be Proven

For every recurrent KV owner/index, inspect: tensor shape, dtype, strides, pointer/storage identity, logical token positions, slot mapping, page/block size, alive/free policy, whether views or aliases share backing memory, and on-rank physical storage. Include one trace for (loop, physical layer, token block) index progression.

General *logical* reference equation if caches for effective depth slots are materialized:

`KV_bytes = batch * seq_length * bytes_per_element * 2 * sum_over_live_cache_slots(n_kv_heads(slot) * head_dim(slot))`.

For uniform geometry `L_eff`: `2 * B * S * L_eff * n_kv_heads * head_dim * dtype_bytes`.

**However:** compute `L_eff` from **actual cache slots**, not simply config `num_hidden_layers * loops`. Reused/frozen/compressed/evicted/circular buffers may change storage, correctness, and access patterns. Multi-GPU head splits may replicate K/V if `n_kv_heads < TP`, or be unsupported for 55 heads. Include prelude/coda caches and any recurrent non-KV state.

Distinct accounts (do not merge them):

1. Theoretical logical full BF16 per-token KV.
2. Actually live unique KV elements/bytes by cache policy.
3. Physical allocated/reserved GPU HBM and allocator fragmentation.
4. Measured GPU memory peak during prefill/decode and CUDA graph capture.
5. Physical bytes copied or transferred (CUPTI/Nsight if available), distinct from logical element counts.
6. Device memory bandwidth/HBM traffic **only if measured**. Never infer bandwidth from a copy kernel's name or logical bytes.

Per-rank decomposition: `weights`, `KV`, `temporary activations`, `graph/paging/workspace`, `other/fragmentation`. Compare predicted vs measured allocation, and explain all important mismatch. Include evidence of cache correctness; no accidental overwrite of previous token's recurrent step KV.

---

## 5. Frozen Workloads and Comparability

**Stage-A calibration:** 8 representative prompts, freeze by metadata before profiling: 3 coding/reasoning, 3 math, 2 long-document / QA. Prompt text hashes + tokenizer revision + required prefix IDs. Suggested held-out 24 when GO signal is strong; do not rerun exhaustive evaluation after a clear kill gate.

For each model independently, choose supported logical context lengths from:

`S in {1024, 4096, 8192, 16384, 32768}`.

**Verify actual checkpoint positional limits and attention semantics before execution.** For Ouro base, 32K may be extrapolated, whereas Thinking claims 32K SFT; Huginn's config `block_size=4096` must not be silently treated as 32K support. If longer context requires model modification, report `CONTEXT_LIMITED`; do not invent data beyond supported length. Use the maximal *honestly supported* comparable subset.

Generated token budget for performance: 64 or 128 deterministic tokens per request, independently configurable; use fixed decode count where possible and a separately reported natural-EOS run for quality. Ensure the measured KV fill sequence matches requested prefix length; prefill with one batch of S and decode N cached positions. Distinguish "32K prompt" from "32K KV capacity".

**Model-specific depth sweeps:** Ouro R=4 primary; optional supported R in {1,2,4} as separate quality-varying axes. Huginn primary fixed step count from its official recommendations (e.g. 16/32) and optional {8,16,32} where valid. NEVER compare Ouro R4 and Huginn R32 as equal compute/depth. Report logical layer invocations/token and quality.

**Concurrent serving:** `concurrency in {1,2,4,8}` if feasible within HBM; count completed requests and errors. For B1, prioritize per-request E2E/TPOT; for B2/B4 report total throughput, p50/p95 latency, tail stalls, and max feasible concurrency. Do not label TP2×DP2 aggregate TPS as a B1 speedup.

---

## 6. Stage A — Native TP1 / TP2 / TP4 Physical Baselines

**Primary aim:** reveal actual crossover between compute, cache capacity, cache bandwidth, and TP collectives without a new method.

On GPUs 4/5/6/7, using compact error-free official native KV policies:

- TP1 on physical GPU4 (unless supported set of GPUs is occupied; document actual hardware and availability).
- TP2 on physical GPUs4,5; optional independent TP2 on GPUs6,7 only to check hardware variability.
- TP4 on physical GPUs4,5,6,7.
- Optional TP2×DP2 for concurrency=2,4 and TP1×DP4 for throughput, **only after valid single-replica capacity tests**. The full 4-GPU count does not automatically mean TP1×DP4 works at long context.

**Required true-TP proof:** rank/UUID mapping, model weight partition, Q/KV head ownership, state cache ownership, actual NCCL calls, no hidden replication when reporting shard ratios, and cross-rank output correctness. TP2/TP4 must actually execute pretrained recurrent steps, not a tensor-shape toy model.

Measurements (paired per same prompt and model/config):

- Cold prefill latency (loaded/warmed engine, no prefix cache hit), prefill peak HBM.
- Warm cached decode per-token TPOT (with separate prefill), B1 E2E, actual output TPS; exact generated length and EOS.
- Loop-by-loop attention time, cache read/update, recurrent-core compute, QKV projections, MLP, TP collective API intervals and rank waiting; retain non-overlap uncertainty.
- Per-GPU `memory_allocated`, `reserved`, engine-block-pool usage, measured KV bytes, graph buffers; OOM threshold at each S and concurrency.
- Timing in `clean`, `low-overhead events`, `heavy profile` lanes. Never add overlapping per-rank kernel service sums into 100% E2E.
- TP overhead: measured NCCL, number of All-Reduce/All-Gather and payload, round-trip, occupancy; compare real measured compute, not just FLOPs proxy.
- Request-level repeat 3 times with counterbalanced arm order when resources permit; keep 8 requests as independent units.

**Stage-A direct questions:**

A1. Does KV bytes/rank follow `1/TP`, or are there replicated/bounded/frozen slots?
A2. At long context, is the actual bottleneck **HBM capacity**, **KV traffic**, **compute**, or **NCCL critical path**?
A3. Is TP4 actually faster for B1, or mostly a capacity-enabler?
A4. Is there an accurate capacity-vs-latency-vs-concurrency Pareto frontier and does it differ between Ouro and Huginn?
A5. Does Huginn cache-sharing (native `compress-s16`) already eliminate most loop-depth KV pressure, at what quality cost, and does the same placement hypothesis still apply?

### Stage A stop rules

- If Ouro TP2/TP4 unsupported or correctness cannot be checked: `RUNTIME_BLOCKED` and no method.
- If at all supportable lengths and request counts TP1/TP2/TP4 native variants cover the memory/latency Pareto frontier with no persistent pressure, `NO_GO_NATIVE_TP_SUFFICIENT`.
- If the only result is the obvious `KV/rank = KV_total/TP` and speedup is merely extra HBM, `GENERIC_TP_ONLY`.
- If measurements show performance differences but root cause is trace perturbation, unsupported contexts, graph OOM, or model-specific bad cache implementation, `TIMING_INVALID` / `GENERIC_RUNTIME_ONLY`.
- If one model is executable but the second blocked, continue a **bounded one-model characterization**, label `PARTIAL_MODEL_COVERAGE`, and avoid generality claims.

---

## 7. Stage B — Loop-aware Placement/Execution *Headroom*, Not a Method Yet

Only if Stage A exposes a material physical bottleneck, examine **decoupling model-weight execution parallelism from KV residency**.

Possible formulations (not assumed improvements):

- Conventional **TP4**: head+weight partitions; per-rank full sequence for its owned KV heads.
- **TP2 compute × a different KV ownership degree** using four GPUs: requires exact K/V transfer, gather or distributed attention, ownership, updates, and synchronization. This is *not* the usual independent TP2×DP2 serving; the latter processes distinct requests.
- **Depth/layer KV placement**: caches of different recurrent depth on different GPUs; because loops execute sequentially, **charge** recurrent hidden-state transfers, KV availability, remote reads, and producer–consumer synchronization; idle GPUs are not free resources.
- **Hybrid head × sequence × depth placement**: only if correct with model-specific masks, head geometry, cache reuse, and large enough context to compensate for collectives. A prior dLLM exact CP prototype was slower because communication/merge cost exceeded local compute; this is a caution, not a measured Ouro/Huginn result.
- **Request-aware memory placement**: different TP degrees/replica layouts across coexisting requests can improve *capacity* or *throughput*; benchmark separately from B1.

**Do not** build KV residual compression; no cross-loop similarity analysis; no training; no speculative decoding; no approximation or cache staleness in the primary experiment.

For the best two conceptual candidate layouts, produce:

1. A legal physical rank placement (exact KV owner, Q owner, weight owner, producer–consumer dependencies).
2. A **memory feasibility table** per rank (not only KV); account for replicated weights and graph workspace.
3. A physical communication ledger: every K/V or hidden-state byte transferred, collective call count, output merge, cache update, metadata, and opportunities for overlap.
4. A calibrated cost model using real per-GPU attention/kernel/collective timings from Stage A; declare uncalibrated portions UNKNOWN.
5. A theoretical / simulation *optimistic* upper bound **clearly separate** from realistically realizable gain.
6. At most one microbenchmark using existing optimized kernels and actual four-GPU communication to test the dominant tradeoff; compare against optimized native TP, not naive PyTorch.

**Strong no-go:** an arrangement seems to save per-rank KV memory but merely moves a comparable or larger set of KV bytes over NVLink each iteration, increasing critical-path time. More aggregate GPU HBM alone is not novel if TP already provides the same benefit.

---

## 8. Conditional Stage C — Minimal Physical Candidate

Execute **only if** all are true:

- a persistent, meaningful workload at supported context and recurrence depth demonstrates native TP's shortfall;
- best candidate (with full communication cost) has a **measured** material headroom, not simply a bytes-oracle;
- implementation is bounded and does not require rebuilding a model engine;
- candidate preserves native inference semantics or a separately quality-gated approximation is explicitly requested later.

Integrate the smallest exact placement/dataflow change into an isolated local runner. Do not reimplement a new kernel framework. Validate cross-rank KV/logits and output parity, graph cache safety, actual memory, physical wall time, p50/p95, and output quality. A positive microbenchmark without clean full-model E2E benefit is `MICRO_ONLY`, not GO.

**No automatic migration or active loop switch** unless its real copy/transfer/coordination ROI is positive; the project's earlier dLLM per-block expert replication lost on cold-copy ROI.

---

## 9. Quantitative Exploration Gates (Pre-registered)

These are screening thresholds, not prior predictions:

- **Gate M (capacity):** native TP layouts have an actual supported long-context or concurrency memory limit; any new scheme raises usable, **quality-equivalent** concurrent capacity by ≥25% **at no more than 10% tail-latency penalty**, or enables a representative workload that all comparable native layouts cannot serve. A trivial TP1 OOM but TP4 success does **not** satisfy novelty.
- **Gate L (latency):** for a representative B1 supported workload, a physically realizable candidate shows ≥10–15% **clean E2E latency** reduction versus best feasible native TP baseline, or enough measured targetable critical-path share to credibly exceed that threshold. The latter merits only `CONDITIONAL` until validated.
- **Gate T (throughput):** with same four GPUs and comparable quality/p95 SLO, ≥20% aggregate tokens/s or completed requests/s versus best feasible `TP2×DP2`, `TP1×DP4`, and `TP4`, **not just TP4**.
- **Gate X (model generality):** the underlying cache/dependency problem is observed in at least two **architecturally different, physically validated** recurrent models, or there is a clearly model-agnostic theorem/algorithm supported by a convincing second model characterization. If Huginn TP impossible, do not award generality based on Ouro-1.4B.
- **Gate Q (correctness):** no change in expected outputs for the exact primary path; official quality benchmark required before claims of model fidelity for any policy-changing option. TP degrees may naturally differ numerically; report reference-parity tolerances and first-difference cause.

If only KV logical bytes/loop trend changes without measured latency, or if native TP solves the problem with minimal cost, verdict `NO-GO`. An optimistic remote-KV oracle without transfer costs is insufficient.

---

## 10. Correctness and Training-Free Constraints

Primary scientific baseline: original pretrained weights and deterministic fixed-depth/cache policy, BF16, correct RoPE, attention masking, sequence positions, EOS, and no output truncation misreported as accuracy.

- Same-config instrumentation OFF/ON should preserve output, cache state, loop count, sampled token order, and logits within frozen tolerance. Distinguish native nondeterminism from interventions.
- TP1 vs TP4 may show different BF16 accumulations. Do not require blanket bitwise identity across TP degrees if the native engine does not guarantee it; compare numerical error, target tokens where deterministic, and benchmark accuracy appropriately.
- Cache-policy comparisons such as Huginn `compress-s16` are **separate** and may alter semantics. Report quality/cost as Pareto points, not as exact alternatives.
- For a small frozen task subset: natural-EOS/full-token-budget questions; optional official GSM8K or HellaSwag/LM-eval with correct prompts/scoring. 64-token bounded probes are **not official benchmark accuracy**.
- Do not assume Huginn `mean_recurrence` or stochastic state initialization yields deterministic restarts. Verify and fix seeds according to official runner and compare same-run interleaved arms.
- No full training, model fine-tuning, learned compressor, speculative branch method, or quantization introduced in primary.

---

## 11. Measurement and Statistical Integrity

- Record machine load, active GPU owners, clock constraints (do not modify clocks), software pins, warmup, graph capture, Nsight perturbation, and HBM free before each arm.
- Separate `clean wall` / `low overhead CUDA events` / `heavy trace`. Prefer event observer overhead ≤5%; if >10%, fine attribution is untrusted.
- Never sum concurrent GPU kernels and label the result E2E percentage. For PP/loop-dependent execution use a dependency-aware timeline with recorded unresolved gaps.
- For B1, use paired request mean/median and p95; for throughput, use actual aggregate tokens generated / wall sec and p95 request latency, not sum of per-request TPS.
- Freeze prompt manifest. Bootstrap at **request level** (≥2000 resamples); repeat-level and loop/layer-level points are dependent.
- Stop OOM safely; label OOM without excluding it from capacity results. Report memory utilization before/after warmup/graph capture separately; do not call weight size a capacity feasibility proof.
- Do not borrow timings from different models, different runtime versions, or the previous Foil-1 small-model EP PoC to infer new hardware outcomes.

---

## 12. Required Plots and Tables

1. **Architecture & cache-policy comparison:** Ouro vs Huginn, actual recurrent modules and KV slot map.
2. **Analytical vs measured KV memory:** bytes per generated token / loop depth / context; per rank TP1/2/4.
3. **GPU HBM composition:** weights, loop-specific KV, graph/workspace, unclassified; TP degree and context.
4. **B1 prefill + TPOT + clean E2E vs context:** TP1/2/4, separate per model.
5. **Loop × component latency heatmap:** attention, recurrent FFN, KV access/update, NCCL, other; all caveats.
6. **Capacity / concurrency vs S:** TP4, TP2×DP2, TP1×DP4 where physically supported; report p95 latency and throughput.
7. **KV access and NCCL pressure:** measured service times and honest critical path bounds.
8. **Proposed KV-vs-weight partition tradeoff:** actual memory/transfer ledger; best-native vs candidate microbenchmark only if Stage B triggered.
9. **Quality / accuracy per runtime and cache policy** for the tested configurations.
10. **GO/NO-GO decision matrix** with each gate and evidence references.

Suggested headline table per model:

| Model | Actual Cache Policy | Context | Depth | Layout | Peak HBM/rank | KV HBM/rank | Prefill ms | TPOT ms | B1 E2E ms | B1 TPS | Max concurrency | Aggregate TPS | p95 latency | Correctness |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

Use `UNSUPPORTED`, `OOM`, `NOT_RUN`, `UNVALIDATED`, and `UNKNOWN` rather than filling hypothetical numbers.

---

## 13. Deliverables and Repository Hygiene

Recommended project folder:

```text
poc_looped_multigpu_kv_placement/
├── scripts/
│   ├── inspect_model_arch.py
│   ├── inspect_cache_contract.py
│   ├── prove_tp_partition.py
│   ├── run_native_layout_sweep.py
│   ├── trace_loop_memory.py
│   ├── benchmark_capacity.py
│   ├── calibrate_cost_model.py
│   ├── optional_physical_kv_microbench.py
│   └── summarize_results.py
├── results/
│   ├── env_and_revisions.json
│   ├── frozen_workload.json
│   ├── architecture_cache_audit.json
│   ├── tp_ownership_proof.json
│   ├── memory_per_rank.csv
│   ├── request_metrics.csv
│   ├── loop_metrics.csv
│   ├── supported_layouts.json
│   ├── capacity_frontier.csv
│   ├── oracle_costs.json
│   ├── correctness.json
│   └── verdict_gates.json
├── figures/
├── architecture_contract.md
├── cache_contract.md
├── prior_overlap.md
├── reproduce.md
└── report.md
```

Suggested branch: `flashvep/looped-multigpu-kv-placement-poc`.

- Preserve unrelated dirty/staged work. No `git reset --hard` / destructive clean.
- Keep model weights and environments outside Git; do not commit tokenizer/model checkpoints or massive profiler traces; keep reproducible links/hashes, compact summaries and exact metadata. If large raw artifacts are necessary, document where they live and their checksums.
- Stage/commit/push **only** this PoC's files. Never claim push success without verification.
- If missing runtime or absent GPU availability, finish **feasibility report** rather than inventing completed experiments.

---

## 14. Final Report Must Answer These Questions

1. What are the *actual* independent physical layer count, recurrence count, cache index, and unique KV slots for Ouro and Huginn?
2. Is `KV_total/TP` valid for each model's **live allocated GPU storage**, and where does it fail (head count, cache reuse, padding, graph memory)?
3. Did TRUE physical TP1, TP2, TP4 run on GPUs 4,5,6,7 for each model? If not, what exact model/runtime limitation blocks them (especially Huginn's 55 heads)?
4. Did native TP4 meaningfully help B1 latency or only accommodate longer contexts/larger concurrency?
5. What are the measured HBM, prefill, TPOT, throughput, and p95 curves as context and recurrent depth change?
6. Is the limiting resource capacity, KV bandwidth, recurrent compute, TP collectives, or something else? Which conclusion is based on actual measurements versus proxy?
7. Which cache policy does each runtime natively use, and does Huginn's optional cache sharing already address the proposed problem?
8. Is there a workload where **best native feasible** TP/DP configuration still has a material, measurable memory–latency/throughput shortfall?
9. Does a legally specified decoupled KV placement save actual physical time after every remote-KV movement, query/result merge, sync, and workspace cost?
10. Can this issue generalize across two different looped architectures without forcing fake equivalence of their runtimes or loop count?
11. What existing research/prior already solves related compression/reuse/batching/distribution?
12. Is a new method justified? **GO / CONDITIONAL / GENERIC_TP_ONLY / GENERIC_RUNTIME_ONLY / MODEL_B_TP_BLOCKED / PARTIAL_MODEL_COVERAGE / NO_GO_NATIVE_TP_SUFFICIENT / RUNTIME_BLOCKED / RESOURCE_BLOCKED / TIMING_INVALID**.
13. If GO, specify **only one** smallest next physical experiment—not a speculative full framework.
14. Report tested GPU UUIDs, pinned model/runtime commits, reproducibility steps, Git branch, commit SHA, and final report path.

**Bottom line:** This PoC is a falsifiable *two-architecture* test of whether recurrence-indexed KV memory creates an important **multi-GPU-specific** placement/execution challenge beyond existing TP. “Looped models use a lot of KV” and “TP4 divides that KV by four” are known facts/observations, not contributions. The PoC succeeds scientifically even if it returns NO-GO with a measured reason.

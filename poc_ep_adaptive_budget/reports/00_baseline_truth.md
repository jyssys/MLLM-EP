# 00 — Baseline and runtime truth

Working contract: `/home/esjung/MLLM-EP-github/poc_flashvep/reports/ep_cost_aware_adaptive_refinement_budget_poc_spec.md`, SHA-256 `7fd3901ee51132eb495ab24a34500a9d799584f0e6f42909fe5e91cd06386f62`.

Primary checkpoint is `inclusionAI/LLaDA2.0-flash`, downloaded revision
`744c3f8c6c8317d2377d6d16d8a3d4be2caef563` at
`/home/esjung/models/LLaDA2.0-flash-744c3f8`. The downloaded config has
SHA-256 `ac35e9dc8313f49b2600da2a8b3ec31c53c01d22a5943ec26c8e21f8e208ab07`:
32 layers, hidden 4096, 256 routed experts, top-8, MoE intermediate 1024.
Runtime checkout HEAD is `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`
with existing uncommitted substrate edits; this PoC does not modify it.

Every task-owned run fixes `CUDA_VISIBLE_DEVICES=4,5,6,7`, logical GPU
0/1/2/3 -> physical 4/5/6/7. Physical UUIDs and launch-time owner/free-memory
checks are in each result's `physical_gpu_inventory.csv` and
`processes_before.csv`. The four cards have NV18 links in the `topology.txt`
and `nvlink.txt` audits. Current dInfer/SGLang log explicitly initializes
DeepEP normal EP size 4; source maps 256 experts linearly into 64 local
experts/rank, partitions dense TP4 rows into true EP4 source shards, then
dispatches to owner ranks and performs reverse combine. The previously
captured [runtime route evidence](../../poc_ep_unmasking/reports/00_baseline_unmasking.md)
documents nonzero remote assignments. A fresh layer-16 trace on physical GPUs
4--7 preserved exact assignment conservation: a 992-row MoE invocation had
7,936 top-k branches, 5,857 remote branches and owner-rank loads
`[2086,1557,2827,1466]`. Rank 0 itself sent branches to all four owners,
proving remote dispatch, owner execution, and reverse combine rather than a
TP-only setting flag. Source: [TRACE_SHAPE.csv](../results/observer_base_gsm32/TRACE_SHAPE.csv).

Baseline sampler is `ThresholdParallelDecoder`, temperature 0, block size
32, threshold 0.9, prefix cache, block diffusion, early stop, submitted 32,
mini-batch 32, gen-request 128, BF16, DeepEP normal. The stock decoder is
**not** fixed-k: it commits all eligible positions above 0.9, with a top-token
fallback when none reach 0.9. All static controls use config=0 plus the exact
config=42 cache/decoder flags, because config=42 silently resets the requested
threshold to 0.9. The all-0.9 intervention hook reproduced 32/32 baseline
answer strings and NFE exactly.

Discovery GSM8K-32 threshold-0.9 runs reproduced 13/32 correct and NFE 102.
Their clean BCTs were 11.251, 7.743, 7.548 and 7.653 s; identical outputs
with large timing variance preclude a one-run speedup claim. GPU memory during
current runs is around 74.5--75.1 GiB/rank. The promoted matched GSM8K-512
baseline completed 16 waves, NFE 1300, and 68/512 correct in each of three
independent restarts; BCTs were 113.864, 92.418 and 100.336 s (median
100.336 s). One GSM8K-512 answer changes score by 0.1953125 pp.

Official GSM8K full test (1,319 examples, same official first 100 prompt/truth
anchor) completed 42 waves with 144/1,319 correct, NFE 3,226 and BCT
227.782 s. One item is 0.075815 pp. Full HumanEval (164 official problems)
completed with 13/164 pass, NFE 672 and BCT 47.155 s. HumanEval mini32
OOMed in a long-code cohort after three of six waves; this failed launch has
no headline score. Both HumanEval comparison policies use the strongest
feasible mini16, while GSM8K uses feasible mini32. Dataset hashes are in
[GSM8K manifest](../data/manifest.json) and
[HumanEval manifest](../data/humaneval_manifest.json).

Important harness confound: the first 32 GSM8K problems scored 13/32 when
evaluated as one 32-request pool, but 5/32 within the sorted 512-request
pool. The benchmark's bucketed generation canvas changes actual generated
length with the ready cohort; examples in the 512 pool generated 111 tokens
where the 32-only pool generated 175. Therefore cross-pool raw score/latency
comparison is invalid. Every policy promotion comparison uses the *same*
dataset, sorting, submitted/mini size and generation request.

Current dense-block runtime keeps physical 32-row blocks until they finish.
An aggressive acceptance can measurably remove complete future NFE, but
removal of finalized rows **within** a remaining forward is only a
post-Epoch-compaction sensitivity, not current measured speedup. The current
PoC does not credit within-forward dead-row compaction. A decision/EP trace
captured 99 real routed waves, but its baseline BCT was 12.037 s versus clean
GSM8K-32 median about 7.65 s (+57%). It is observer-only geometry evidence,
not clean stage/E2E timing. The earlier same-substrate
[routed expert pie](../../poc_refinegemm/reports/01_operator_breakdown.md)
estimated about 29.5% of request E2E as observer-attributed upper-bound mass;
this is not a measured removable pie for the present decoder policy.

# MLLM MoE EP Project Backup Summary

Backup date: 2026-06-29

This folder is a lightweight backup of `/home/work/euisoo.jung/mllm-moe-ep`.
Large model weights, benchmark datasets, HuggingFace cache, and generated Python caches were intentionally excluded.

## Backup Scope

Included:

- Project source code: `calib/`, `hooks/`, `measure/`, `method1/`, `method2/`, `pipeline/`, `scripts/`, `tests/`
- Project docs and specs: `docs/`, `spec_phase1.md`, `spec_phase2a.md`, `spec_phase2b1.md`
- Environment metadata: `env/`, `pytest.ini`, helper Python files
- Small experiment outputs and figures under `outputs/`
- Small external reference repos under `external/`

Excluded:

- `models/`: Qwen3-VL-30B-A3B-Instruct weights, about 58GB
- `data/`: benchmark data, HF cache, ShareGPT4V assets, about 15GB
- `back-up/`: excluded from itself
- `__pycache__/`, `*.pyc`, and common local cache folders

## Project Goal

The project studies straggler reduction for Expert Parallelism inference in MoE-based multimodal LLMs, focusing on Qwen3-VL-30B-A3B-Instruct.
The working hypothesis is that vision-heavy multimodal prompts route many more tokens to vision-preferred experts, creating hot experts and hot EP ranks.

The project has three main lines so far:

1. Phase 1 CPU-only logic and dummy tests.
2. Phase 2A vLLM 8-way EP profiling and motivation measurements.
3. Phase 2B1 layer-wise expert placement experiments.

## Phase 1 Status

Phase 1 implemented pure-PyTorch, framework-independent logic and tests:

- `method1/placement.py`: placement utilities and LPT-style balancing.
- `method2/importance.py`: raw cross-attention importance path; de-RoPE and CLS remain Phase 2 TODOs.
- `method2/selection.py`: candidate token selection logic.
- `method2/merge.py`: routing-preserving merge logic.
- `method2/cap.py`: cap policy to stop merging once straggler load reaches the target.
- `calib/collect_stats.py`: CPU dummy calibration statistics, inspired by MODE-style routing frequency collection.
- `hooks/register_hooks.py`, `method2/derope.py`, `pipeline/ep_integration.py`: Phase 2 interface stubs or integration boundaries.

Tests were added under `tests/` for placement, importance, selection, merge, cap, calibration, pipeline, and Phase 2 interfaces.

Useful docs:

- `docs/model_arch.md`
- `docs/calibration_mode_report.md`
- `docs/phase2_stub_report.md`

## Phase 2A Status

DeepSpeed EP over the HF model was blocked because the model was effectively replicated per rank.
The project switched to vLLM native expert parallelism.

Established substrate:

- vLLM native 8-way EP
- `tensor_parallel_size=8`
- `enable_expert_parallel=True`
- `expert_placement_strategy="linear"`
- `all2all_backend="allgather_reducescatter"`
- `enable_return_routed_experts=True`
- `CompletionOutput.routed_experts` shape: `[seq, layer, topk]`
- Expert count: 128 experts, linear rank mapping `rank = expert_id // 16`
- KV cache bounded during routed-expert capture

Important limitation:

- vLLM routed capture exposes expert ids, but not router gating softmax scores.
- Therefore P1/P3 routing-frequency and modality-distribution profiling are available, while P2 gating-score heatmaps remain TODO unless a deeper vLLM hook is added.

Key Phase 2A outputs:

- `outputs/motivation/token_modality_ratio.png`
- `outputs/motivation/ep_straggler_rank.png`
- `outputs/motivation/ep_straggler_expert.png`
- `outputs/calibration/trace_freq.png`
- `outputs/calibration/dist_vision.npy`
- `outputs/calibration/dist_text.npy`
- `outputs/calibration/dist_diff.png`
- `outputs/calibration/dist_table.md`

Useful docs:

- `docs/vllm_ep_notes.md`
- `docs/phase2a_report.md`
- `docs/ep_sim_validation.md`

## Phase 2B1 Status

The main Phase 2B1 target was layer-wise modality-aware expert placement.
vLLM was patched at the expert-map level, without modifying fused MoE kernels.

Relevant files:

- `vllm_custom_placement.py`: injects per-layer expert-to-rank maps into vLLM FusedMoE instances.
- `scripts/build_layerwise_placement.py`: builds placement maps from calibration distributions.
- `scripts/optimize_tail_placement.py`: optimizes layer-wise placement using batch/layer/expert token counts.
- `scripts/vllm_phase2b1_compare.py`: runs As-Is/To-Be comparisons.
- `vllm_moe_timing.py`: CUDA-event timing wrapper around vLLM FusedMoE forward.
- `scripts/summarize_moe_cuda_timing.py`: summarizes MoE-only CUDA timing traces.

Layer-wise placement evolved from simple modality-balanced placement to a tail objective:

```text
R[b,l,r] = sum_e token_count[b,l,e] * 1[m_l(e)=r]
imbalance[b,l] = max_r R[b,l,r] / mean_r R[b,l,r]
objective = mean + 0.5 * p95 + 0.2 * max + capacity/rank-total penalties
```

This objective directly targets batch-layer straggler tails, so it is better aligned with prefill critical-path behavior than an average-only expert distribution objective.

## Latest Batch-32 Findings

The latest important experiment used batch size 32 and main-data-derived calibration.
It compared As-Is linear placement against To-Be tail-optimized layer-wise placement.

Token scale:

- 6 batches
- 192 total samples
- 313,440 total prefill tokens
- Mean prefill tokens per batch: 52,240
- Mean prefill tokens per sample: 1,632.5
- Total routed assignments: 120,360,960

Offline placement objective:

- Linear p95 batch-layer imbalance: about `1.5006x`
- Tail-optimized p95 batch-layer imbalance: about `1.0345x`
- Linear max batch-layer imbalance: about `1.6309x`
- Tail-optimized max batch-layer imbalance: about `1.0531x`

Clean vLLM wall-clock run:

- As-Is total elapsed: `42.3955s`
- To-Be total elapsed: `42.8201s`
- Wall-clock change: about `-1.00%` speedup, meaning To-Be was slightly slower
- Mean TTFT: `3.1873s -> 3.3593s`
- Mean scheduled-to-first-token: `1.1335s -> 1.2558s`

Observed routed-load improvement:

- Batch-layer mean imbalance: `1.2673x -> 1.0196x`
- Batch-layer p95 imbalance: `1.5033x -> 1.0352x`
- Batch-layer max imbalance: `1.6309x -> 1.0536x`
- Layer-total rank p95 imbalance: `1.2917x -> 1.0161x`

MoE-only CUDA timing:

- As-Is MoE critical path: `4588.74ms`
- To-Be MoE critical path: `4159.88ms`
- MoE critical-path reduction: about `9.35%`

Interpretation:

- Layer-wise placement clearly reduces routed-load stragglers.
- It also reduces measured MoE-only critical-path time.
- However, the full vLLM wall/TTFT path did not improve in the clean batch-32 run.
- One likely reason is Amdahl's law: MoE-only critical time was only about `10.8%` of total wall time in the measured run, so even a 9.35% MoE improvement has only about a 1% ideal E2E gain ceiling before scheduler, preprocessing, attention, KV/cache, communication, and noise.

Useful docs:

- `docs/phase2b1_tail_placement_report.md`
- `docs/phase2b1_maincalib_timing_report.md`
- `docs/phase2b1_b32_moe_timing_report.md`
- `docs/phase2b1_latency_gap_analysis.md`

Useful outputs:

- `outputs/main_calib_tail_b32/tail_optimization_summary.json`
- `outputs/main_calib_tail_b32/offline_layer_imbalance.png`
- `outputs/asis_tobe_b32_final/summary_b32_final.json`
- `outputs/asis_tobe_b32_final/summary_b32_final.png`
- `outputs/asis_tobe_b32_final/timing_b32_clean.png`
- `outputs/asis_tobe_b32_moetiming/moe_cuda_timing_summary.json`
- `outputs/asis_tobe_b32_moetiming/moe_cuda_summary.png`

## Current Research Interpretation

The current defensible claim is:

> Layer-wise placement substantially reduces routed-load imbalance and improves MoE-only critical-path timing.

The current not-yet-defensible claim is:

> Layer-wise placement alone reduces full prefill TTFT or end-to-end latency.

My working interpretation is that Method 1 placement is a useful substrate but may not be enough as a standalone latency method in vLLM's full serving path.
Visible end-to-end gains likely require one or more of:

- more precise timing windows around only measured prefill batches,
- communication-aware placement terms,
- larger or more compute-bound batch regimes,
- Method 2 merge/cap to reduce actual work rather than only moving expert ownership,
- integration with online scheduling or overlap mechanisms.

## Suggested Resume Checklist

After the server restarts:

1. Recreate or activate the environment described in `env/`.
2. Restore/download Qwen3-VL weights into `models/Qwen3-VL-30B-A3B-Instruct/`.
3. Restore benchmark/calibration data into `data/`.
4. Run `pytest` to confirm the lightweight code path still works.
5. For vLLM experiments, first run the EP sanity path before long profiling jobs.
6. Treat `docs/phase2b1_latency_gap_analysis.md` as the latest high-level interpretation checkpoint.


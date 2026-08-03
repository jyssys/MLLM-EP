# Phase 2-A Report

Date: 2026-06-25

## Status

vLLM native 8-way Expert Parallelism for Qwen3-VL-30B-A3B-Instruct is working.
The DeepSpeed path replicated the full model on every rank, but the vLLM path
loads the MoE experts sharded across 8 GPUs and runs multimodal generation with
routed-expert capture enabled.

The single-GPU routing comparison is kept as a cautionary cross-check, not as a
blocker. Its mismatch is expected because it compares a 1-GPU arithmetic path
against vLLM's 8-rank EP execution path. The follow-up consistency audit can be
done separately with another agent.

## Step 1: vLLM EP Investigation

Documented in `docs/vllm_ep_notes.md`.

Key findings:

- installed vLLM: `0.20.0+cu129`
- Qwen3-VL-MoE implementation: `Qwen3VLMoeForConditionalGeneration` using
  `Qwen3MoeSparseMoeBlock` and `FusedMoE`
- Qwen3-VL-30B-A3B-Instruct MoE shape: 48 layers, 128 experts, top-k 8
- working EP settings: `tensor_parallel_size=8`,
  `enable_expert_parallel=True`, `expert_placement_strategy="linear"`,
  `all2all_backend="allgather_reducescatter"`
- linear placement gives 16 experts per EP rank:
  `0-15`, `16-31`, ..., `112-127`
- routed expert extraction works through vLLM
  `enable_return_routed_experts=True`, returning
  `CompletionOutput.routed_experts` with shape `[seq, layer, topk]`

Important runtime note: routed-experts capture needs bounded KV cache memory.
With automatic KV sizing, shared-memory capture can become multi-GB. The working
sanity setting was `kv_cache_memory_bytes=1073741824`.

## Step 2: 8-Way EP Sanity

Output: `outputs/vllm_ep_sanity.json`

Result: passed.

Evidence:

- vLLM worker names used TP/EP ranks, e.g. `Worker_TP0_EP0`
- log showed `EP Rank 0/8`
- log showed local/global experts `16/128`
- log showed EP weight filtering: loading `16/128` experts per EP rank
- model memory after load: `11,825 MiB` on each of 8 GPUs
- earlier failed DeepSpeed replica baseline: `57.94 GB` per rank
- dummy multimodal prompt: 79 prompt tokens, 64 image tokens
- routed-experts shape: `[79, 48, 8]`
- routed-experts ids: min `0`, max `127`

This establishes that vLLM is not replicating the full packed MoE on every GPU.

## Step 3: End-To-End Generation

Output: `outputs/vllm_ep_generation_check.json`

Result: passed.

Three real parquet-backed image-text samples were run through vLLM EP:

| sample | prompt tokens | routed shape | short output |
| --- | ---: | --- | --- |
| ChartQA | 540 | `[542, 48, 8]` | `12` |
| TextVQA | 699 | `[746, 48, 8]` | identified `Dakota Digital` |
| MMMU | 279 | `[326, 48, 8]` | answered `$6` with a short reason |

This confirms vLLM EP generation executes on actual multimodal samples and
returns routed experts.

## Smoke Accuracy Check

Output: `outputs/accuracy_smoke/chartqa20_vllm_ep.json`

Result: passed as a small operational sanity check, not an official benchmark.

I ran the first 20 examples from the local ChartQA test parquet through the same
vLLM native EP setup:

- `tensor_parallel_size=8`
- `enable_expert_parallel=True`
- `expert_placement_strategy="linear"`
- `all2all_backend="allgather_reducescatter"`
- `enable_ep_weight_filter=True`

The smoke score was `13 / 20 = 65.0%`. This uses a lightweight exact/normalized
answer matcher with numeric tolerance for scalar values; four-digit year answers
require exact match. Because this is only 20 examples, it should be treated as a
quick end-to-end health check rather than a reportable ChartQA accuracy number.

Memory after loading stayed at about `12,095 MiB` per GPU, still far below the
DeepSpeed full-replica baseline. The smoke run also completed without OOM or
NCCL runtime failure.

## Routing Cross-Check

Documented in `docs/ep_sim_validation.md`.

The cross-check compared:

- actual vLLM 8-way EP: `tensor_parallel_size=8`,
  `enable_expert_parallel=True`
- 1-GPU full model: `tensor_parallel_size=1`
- simulated rank mapping: `rank = expert_id // 16`

Both paths returned `[79, 48, 8]` routed-expert arrays, but exact routing did not
match:

- routing entry mismatches: `16,224 / 30,336`
- expert-load L1 difference: `898`
- rank-load L1 difference: `106`
- EP rank load: `[3650, 3635, 4085, 3660, 4023, 3582, 4019, 3682]`
- single+linear-map rank load:
  `[3653, 3634, 4082, 3657, 4073, 3575, 4017, 3645]`

Interpretation: this does not invalidate vLLM EP. The comparison changes the
router input arithmetic path as well as the physical MoE placement. Small BF16,
kernel, and parallel-reduction differences before the router can change top-k
membership for close expert logits. Rank-level load remained very close for the
dummy input, but single-GPU routing should be treated as an approximation rather
than an exact oracle for vLLM EP.

## Motivation Measurements

Measurement script: `scripts/vllm_phase2a_profile.py motivation`

Data:

- MMMU: 64 samples, multi-image samples first
- ChartQA: 64 high-resolution chart samples
- TextVQA: 64 high-resolution OCR samples
- MMBench: 64 visual QA contrast samples

All counts are prefill-only. `CompletionOutput.routed_experts` was sliced to the
prompt length so decode tokens are excluded. Vision tokens were identified by
Qwen image/video token ids (`151655`, `151656`).

### M1 Token Modality Ratio

Outputs:

- `outputs/motivation/token_modality_ratio.json`
- `outputs/motivation/token_modality_ratio.png`

Result: vision tokens dominate the prefill router input.

| split | samples | vision tokens | text/control tokens | mean vision ratio |
| --- | ---: | ---: | ---: | ---: |
| all | 256 | 287,263 | 15,013 | 90.3% |
| ChartQA | 64 | 63,423 | 2,066 | 96.8% |
| MMMU | 64 | 127,392 | 7,064 | 86.1% |
| TextVQA | 64 | 80,064 | 1,721 | 97.5% |
| MMBench | 64 | 16,384 | 4,162 | 80.8% |

The median per-input vision ratio was `96.6%`, and the p90 was `97.6%`.

### M2 EP Straggler Evidence

Outputs:

- `outputs/motivation/ep_straggler_rank.json`
- `outputs/motivation/ep_straggler_rank.png`
- `outputs/motivation/ep_straggler_expert.json`
- `outputs/motivation/ep_straggler_expert.png`

Method: expert ids were mapped to ranks with the validated linear placement
`rank = expert_id // 16`. Counts include top-k multiplicity. Because aggregating
over all samples and all 48 layers smooths away the actual per-iteration
straggler, the plotted figure uses the batch/layer with the largest rank
max/mean imbalance. The aggregate all-layer metrics are still saved in JSON.

Plotted scope:

- batch: samples `208-215`
- dataset: MMBench
- layer: `20`

Hot-rank result:

- rank max/mean load imbalance: `1.95x`
- hot rank: `1`
- hot rank load: `4,752` routed assignments
- scope mean vision ratio: `84.2%`
- hot rank vision ratio: `87.7%`

Hot-expert result:

- expert max/mean load imbalance: `6.74x`
- hot expert: `23`
- hot expert load: `1,025` routed assignments
- hot expert vision ratio: `89.8%`

This supports the motivation claim: in the actual vLLM EP run, the hot
rank/expert in the straggler scope is more vision-heavy than the batch average.
The all-layer aggregate rank imbalance was only `1.04x`, which confirms why the
straggler should be analyzed at batch/layer granularity rather than averaged
over the whole profiling set.

## Calibration Profiling

Measurement script: `scripts/vllm_phase2a_profile.py calibration`

Data: ShareGPT4V 512 local image-text pairs from `data/sharegpt4v_512/`.

All counts are prefill-only and use actual vLLM EP routed expert ids.

### P1 Trace Frequency

Outputs:

- `outputs/calibration/trace_freq.npy`
- `outputs/calibration/trace_freq.png`

Result:

- shape: `(48, 128)`
- total routed assignments: `56,598,528`
- values are layer-by-expert top-8 selection counts

### P2 Gating Score

Outputs:

- `outputs/calibration/gating_score_todo.md`
- `outputs/calibration/gating_score_todo.json`

Status: TODO, not blocked.

vLLM 0.20 exposes `CompletionOutput.routed_experts`, but the built-in capture
stores only `topk_ids` via `RoutedExpertsCapturer.capture(layer_id, topk_ids)`.
Router softmax values / `topk_weights` exist inside the fused MoE routing path
but are not exported through the public output API. Per the Phase 2-A rule, I
did not patch fused kernels or alter model execution.

### P3 Vision/Text Expert Distribution

Outputs:

- `outputs/calibration/dist_vision.npy`
- `outputs/calibration/dist_text.npy`
- `outputs/calibration/dist_diff.npy`
- `outputs/calibration/dist_diff.png`
- `outputs/calibration/dist_vision.png`
- `outputs/calibration/dist_text.png`
- `outputs/calibration/dist_table.md`

Result: vision and text tokens prefer meaningfully different expert subsets.

- matrix shapes: `(48, 128)` for both `P_vis` and `P_txt`
- each layer is normalized over experts, row sum = `1`
- calibration vision ratio: `93.1%`
- mean `|P_vis - P_txt|`: `0.00948`
- mean total variation distance: `0.607`
- max total variation distance: `0.734` at layer `9`
- top TV-distance layers: `9, 11, 13, 12, 8, 23, 25, 7`

Example from `dist_table.md`: at layer `9`, the top vision experts are
`E47, E19, E31, E74, E3`, while the top text experts are
`E49, E69, E10, E114, E122`. This is the direct empirical basis for the next
phase's modality-aware placement work.

## Current Conclusion

Use vLLM native 8-way EP as the measurement substrate for the next Phase 2-A
and Phase 2-B work. The implementation and measurements are sufficiently
established by:

- expert weight sharding evidence (`16/128` experts per rank)
- rank memory reduction versus full replication
- successful multimodal generation
- successful routed-expert capture with full expert id range
- M1 evidence that prefill router input is vision-dominated (`90.3%` mean)
- M2 evidence that hot EP rank/expert scopes are vision-heavy stragglers
- P3 evidence that vision/text tokens select different expert distributions

The exact equivalence question can be audited separately, but the working path
for profiling should be the actual vLLM EP run, not single-GPU simulation.

No Method 1 placement, Method 2 merge/cap, de-RoPE/CLS, or speedup measurement
has been applied. Those remain TODO(Phase2B).

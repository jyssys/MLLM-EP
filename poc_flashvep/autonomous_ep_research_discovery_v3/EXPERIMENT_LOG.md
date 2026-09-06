# Experiment log

All fresh measurements used physical GPUs 1–4 only. The model was loaded from
the local Qwen3-VL snapshot with BF16, vLLM 0.20 V1, TP2/DP2/EP4, DeepEP
high-throughput, eager mode, prefix cache disabled and DBO disabled. The hook
is read-only and records CUDA-event stage intervals plus route statistics.

| UTC/KST window | Run | Purpose | Live result | Follow-up |
|---|---|---|---|---|
| 17:23–17:29 | `live_mixed_c8` | varied real-image/text continuous batch | 107,616 logical warm+measured rows; measured T_MoE p50 1.16ms; decode/ prefill stage mix; dispatch max 1.56s | compare fixed modality and concurrency |
| 17:35–17:38 | `live_high_c16` | high-volume varied batch | measured prefill T_MoE p50 1.19ms; p99 5.63ms; expert share 38.6% | load regime evidence |
| 17:43–17:47 | `live_text_c8` | fixed text control | p50 E2E 656.0ms; prefill T_MoE p50 1.17ms; expert 31.0% | fixed c16/c2 |
| 17:49–17:53 | `live_vision_hi_c8` | fixed high-resolution image | p50 E2E 757.8ms; T_MoE p50 1.22ms; expert 42.2% | modality composition comparison |
| 17:53–17:57 | `live_text_c16` | fixed text high concurrency | p50 E2E 773.1ms; T_MoE p50 1.40ms; dispatch 28.4%, wait 14.6% | concurrency curve |
| 17:58–18:03 | `live_mixed_c2` | low concurrency varied control | p50 E2E 285.6ms; T_MoE p50 1.21ms | low-load anchor |
| 18:05–pending | `live_text_c2` | fixed text low-concurrency control | trace completed after extended decode; atlas generated after shutdown | c2/c8/c16 comparison |
| 18:34–18:48 | `live_mixed_c8_mbt4096` | token-budget causal control | same M bins as MBT8192 but T_MoE p50 2.055ms vs 1.164ms; E2E p50 1,278ms vs 901ms | classify as trivial configuration effect |
| 18:49–18:53 | `live_long_text_c8` | long-prefill volume probe | M=284/852, T_MoE p50 2.012ms, dispatch 45.9%, expert 25.4%, wait 13.1%; E2E p50 965ms | no independent high-mass variable |

Known errors were bounded: an initial text c8 launch used a wrong cache path
and exited before model load; it was rerun with the validated snapshot. The
repeated `cpuinfo` usage thread warning is outside the CUDA serving path and
did not change exit status. No unsupported backend or GPU was used.

| 19:45–19:56 | `live_text_c8_telemetry` | 40-wave text target after default vision warmup; GPU clock telemetry | first 31 waves T_MoE ~2.3–2.7ms and wait ~0.46ms; waves 32–39 ~1.17ms/~0.02ms; E2E wave p50 falls ~1.23–1.33s to ~0.83–0.89s | shape-matched warmup control |
| 19:58–20:01 | `live_text_c8_warmup_text` | text target after text warmup | all 8 waves ~1.16ms T_MoE, wait ~0.018ms; E2E p50 ~843ms | transition test |
| 20:02–20:05 | `live_text_to_vision_c8` | high-res vision target after text warmup | 8 waves T_MoE ~2.0–2.5ms, wait ~0.33ms; E2E p50 1,350ms | high-res vision matched warmup |
| 20:06–20:09 | `live_alternating_text_vision_c8` | alternate text/high-res vision within one worker | all waves ~2.12ms, wait ~0.49ms | text-only shape alternation control |
| 20:10–20:12 | `live_alternating_text_shapes_c8` | alternate two text shapes | waves 0–6 1.20–1.25ms, wait .02–.04ms; final wave tail | modality boundary, not shape count |
| 20:13–20:15 | `live_qwen3_dense_c8` | optional generic Qwen3-30B-A3B fixed text | T_MoE p50 2.15ms, wait .44ms; DeepEP active | replicate generic model |
| 20:16–20:18 | `live_qwen3_dense_c8_rep2` | generic fixed-text replication | T_MoE p50 2.11ms, wait .42ms; consistent | model state differs from VL control; no claim without matched protocol |
| 20:21–20:24 | `live_vision_hi_c8_warmup_hi` | high-res vision after matching high-res warmup | T_MoE p50 1.20ms, wait .028ms; E2E p50 918ms | C1c becomes shape-state diagnostic/trivial |
| 20:24–20:26 | `live_deepep_ll_text_c4_retry` | low-latency backend feasibility | startup failed at `assert nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2`; no measurements | backend-specific branch closed as unsupported in current config |
| 20:32–20:34 | `live_alternating_vision_shapes_c8` | alternate 448px/896px vision shapes | T_MoE p50 1.16ms, wait .029ms; E2E p50 861ms | confirms text↔vision boundary, not shape alternation alone |
| 20:36–20:41 | `live_text_c8_warmup_text_long` | long matched text control | 32 waves / 12,288 logical rows; T_MoE p50 1.175ms, p99 4.41ms; wait p50 .018ms throughout | persistent baseline; closes C1c as preconditioning effect |

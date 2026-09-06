# Raw trace manifest

This manifest is a compact index for the self-contained raw traces in this
result root.  Large `invocations.jsonl`, `stages.jsonl`, and route payloads are
kept in place but are intentionally not duplicated in the compact atlas.

## Environment

- Model: Qwen3-VL-30B-A3B-Instruct (snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`)
- Runtime: vLLM 0.20.0 V1, BF16, eager, DBO off, prefix cache off
- Topology: TP2 / DP2 / EP4 / PP1; DeepEP high-throughput; communication SMS 20
- Visible devices: `CUDA_VISIBLE_DEVICES=1,2,3,4` (physical GPUs 1, 2, 3, 4 only)
- Observer: local NVTX/CUDA-event hook; warmup rows excluded with `active_wave`;
  TP/EP worker rows collapsed by DP/local invocation/layer/phase using the
  maximum same-DP critical span.

## Completed traces used by the atlas

| Trace | Purpose / control | Status |
|---|---|---|
| `live_mixed_c2`, `live_mixed_c4`, `live_mixed_c8` | real multimodal continuous-batch load regimes | complete |
| `live_mixed_c8_mbt4096` | max-batched-token budget control against MBT8192 | complete |
| `live_high_c16` | high-concurrency multimodal regime | complete |
| `live_long_text_c8` | long text control | complete |
| `live_text_c4`, `live_text_c8`, `live_text_c8_rep2`, `live_text_c16` | text concurrency/repetition controls | complete |
| `live_vision_hi_c8` | fixed high-resolution vision control | complete |
| `live_text_c8_telemetry` | clock/telemetry-tagged text run | complete |
| `live_text_c8_warmup_text`, `live_text_c8_warmup_text_long` | text shape-matched warmup controls | complete |
| `live_text_to_vision_c8` | text warmup → high-resolution vision transition | complete |
| `live_alternating_text_vision_c8` | cross-modality alternation control | complete |
| `live_alternating_text_shapes_c8` | text-only shape alternation control | complete |
| `live_vision_hi_c8_warmup_hi` | vision shape-matched warmup control | complete |
| `live_alternating_vision_shapes_c8` | vision-only shape alternation control | complete |
| `live_text_to_vision_c8_telemetry` | independent telemetry-tagged transition replication | complete |
| `live_qwen3_dense_c8`, `live_qwen3_dense_c8_rep2` | optional generic Qwen3 dense-MoE controls; kept separate | complete |

The primary compact atlas contains 21 Qwen3-VL traces and 91,776 collapsed
logical rows.  The generic Qwen3 controls are reported only as a separate
cross-model sanity check and are not pooled into Qwen3-VL conclusions.

## Bounded failures / exclusions

| Trace | Reason excluded from headline atlas |
|---|---|
| `live_text_c2` | aborted/hung after partial collection |
| `live_text_c8_rep3` | aborted/hung; partial diagnostic only |
| `live_text_c8_rankcv` | deadlocked before measured waves |
| `live_text_c4_decode` | aborted during warmup |
| `live_deepep_ll_text_c4`, `live_deepep_ll_text_c4_retry` | low-latency backend assertion: `nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2` |

Failures remain on disk with their driver status or compact logs so that a
future rerun can distinguish unsupported configuration from missing data.

## Analysis entry points

```bash
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  poc_flashvep/autonomous_ep_research_discovery_v3/make_atlas.py
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  poc_flashvep/autonomous_ep_research_discovery_v3/mine_residuals.py
/home/esjung/.venvs/sonic-ep-poc/bin/python \
  poc_flashvep/autonomous_ep_research_discovery_v3/plot_atlas.py
```

Compact derived artifacts are under `discovery_atlas/`; the raw trace files
are retained for audit and are not required to reproduce the headline tables.

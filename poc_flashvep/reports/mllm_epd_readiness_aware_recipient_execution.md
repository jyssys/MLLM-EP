# Readiness-aware recipient execution

The previous Qwen2.5-VL H200 `image-ready` epoch has an exact source join to all 24 original Very-long requests and all 873 image occurrences. For each request, the captured E response, P response and client prompt token IDs match the frozen original processor tensor. The observed E manifest and actual P consumer manifest share an exact SHA-256. Each ordered E event also matches one actual P CUDA IPC clone with the same producer, original image identifier and complete embedding row count. The current geometry audit supplies the original placeholder intervals and verifies each closing vision delimiter. [Trace reducer](../mllm_epd_modality_aware_idle_execution_discovery/readiness_prefix.py) · [Raw-derived results and input hashes](../mllm_epd_modality_aware_idle_execution_discovery/results/readiness_prefix.json)

## Original rows available under a conservative transport bound

For each image, the usable lower time bound is the recorded producer event lower bound. The upper time bound is the later of CUDA event completion observation and host publication return. For an original prefix through image *k*, use the latest upper bound among its first *k* images. For the full request, use the latest lower bound among all images. Their difference is a lower bound on the prefix lead over full-request **transport availability**. The last admissible cut includes the original closing vision delimiter. Earlier ready items cannot be skipped or reordered. [Producer event and manifest semantics](../mllm_epd_prefill_heavy_modality_discovery/image_ready.py) · [Actual P consume contract](../mllm_epd_prefill_heavy_token_split/processor_handoff.py)

| Guaranteed lead before all-image transport availability | Requests with an image-complete prefix | Total contiguous original LM rows | Rows per qualifying request, min / median / max | Most complete images in one prefix |
| --- | ---: | ---: | ---: | ---: |
| ≥5 ms | 10/24 | 196,519 | 4,835 / 21,152 / 24,481 | 56 |
| ≥10 ms | 9/24 | 170,266 | 11,410 / 19,360 / 22,806 | 56 |
| ≥25 ms | 7/24 | 94,908 | 7,739 / 12,532 / 18,222 | 41 |
| ≥50 ms | 6/24 | 38,090 | 1,152 / 7,548 / 11,003 | 27 |

Each row sums the largest qualifying prefix per request. The thresholds reuse the same requests and overlap. These are `DIAGNOSTIC` trace-derived bounds from the prior opt-in observer. The archived epoch contains all matched per-image witness rows but no terminal observer receipt. Native P submission followed the whole-request E acknowledgement in all 24 cases. No early consumer clone, incremental LM execution, deadline-safe D time or task-level latency was measured. Available prefix rows alone give no estimate of the time required to acquire their embeddings and advance causal KV state.

A recipient-policy experiment would feed the exact same early causal frontier to an earliest-causal P control and a P-or-D policy, then measure successful completed LM work, copy/wait costs, Decode interference and TTFT. The live candidate is `NOT_RUN`. The ready frontier itself has a verified early-P precedent in RServe and public gLLM. [Prior implementation comparisons](mllm_epd_idle_execution_prior_work.md)

## Reproduction

```bash
study=poc_flashvep/mllm_epd_modality_aware_idle_execution_discovery
CUDA_VISIBLE_DEVICES='' ../mllm_epd_runtime_env/bin/python \
  -m poc_flashvep.mllm_epd_modality_aware_idle_execution_discovery.readiness_prefix \
  --epoch ../h200_epd_modality_runs/image-ready \
  --manifest ../h200_idle_execution_runs/dataset/primary_very_long_n24.json \
  --geometry "$study/results/vision_work_geometry.json" \
  --out "$study/results/readiness_prefix.json"
```

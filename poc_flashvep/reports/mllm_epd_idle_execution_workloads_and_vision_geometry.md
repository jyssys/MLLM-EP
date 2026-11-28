# Idle execution workload contracts and vision window geometry

## Frozen workloads

The primary Very-long N24 uses the exact previous serving manifest, including the original prompt order and all 873 image occurrences. The median input has 24,789.5 LM rows and 42.5 images. Against the Main400 originals, the payloads match after media and processor metadata locations are normalized. The actual prompt SHA-256, encoded image bytes, and token metadata bytes match their recorded digests for every primary and held-out row. [Cohort and SHA-256 receipts](../mllm_epd_modality_aware_idle_execution_discovery/results/cohort_receipts.json)

| Primary census | Unique original requests | Last planned arrival at QPS 1.465 | Output contract |
|---|---:|---:|---|
| N24 | 24 | 10.206 s | Fixed16 |
| N64 | 24 | 39.135 s | Fixed16 |
| N128 | 24 | 73.229 s | Fixed16 |
| N256 | 24 | 147.463 s | Fixed16 |

Each census uses arrival seed 2026112809, single-route open-loop Poisson offsets, and the original N24 roster cycled in order with separate request IDs. Both prior N24 and prior N64 native arrival traces match the corresponding new planned rosters and offsets exactly. N128 and N256 extend that deterministic series. Repeated-census counts measure persistence at the chosen load and do not add new documents to the workload. [Freeze code](../mllm_epd_modality_aware_idle_execution_discovery/cohorts.py)

The independent Natural EOS contract uses all eligible original Main400 requests with an LM input of 16,384–32,512 tokens and a prior successful native EOS response of 1–24 tokens with finish reason `stop`. It excludes primary request IDs, prompt hashes, known source document IDs and any shared original image byte hash. No new candidate outcome, latency or generated text is used in selection. The pre-existing native census is joined by original request ID, dataset, prompt hash and input length. The serving contract for these held-out requests preserves the original prompts and images and permits Natural EOS. This historical short-output selection describes a conditional workload, since new native or candidate generations can emit a different length.

| Held-out manifest | Original rows | MP-DocVQA / MMLongBench-Doc | Median LM rows | Median images | Prior native emitted tokens, min / median / max |
|---|---:|---:|---:|---:|---|
| Seeded first N24 | 24 | 11 / 13 | 21,085 | 25.5 | 2 / 15 / 24 |
| Full eligibility set | 34 | 17 / 17 | 21,481 | 15.5 | 2 / 15 / 24 |

The seeded order is determined by SHA-256 of the original request ID and dataset seed 2026112807. The N24 is a prefix of the N34. Primary and held-out manifests with actual file paths remain under the external run root. [Cohort receipts](../mllm_epd_modality_aware_idle_execution_discovery/results/cohort_receipts.json)

## Vision execution geometry

For the pinned Qwen2.5-VL configuration, vision blocks 0–6 operate independently on local windows. Block 7 is the first block with full-image attention. Its input requires the full original image's window-ordered patch activations and original position metadata. Full-image attention repeats at blocks 15, 23 and 31. Patches have width 14 pixels, the spatial merger groups 2×2 patches, and a local window covers at most 4×4 merged rows or 64 patches. A rectangular edge window can have fewer patches. The source attention offsets omit synthetic padding, so the audit counts actual patches.

| Exact geometry | Primary N24 | Held-out N34 |
|---|---:|---:|
| Image occurrences | 873 | 686 |
| Distinct processor grids | 9 | 32 |
| Nonempty local-window occurrences | 42,935 | 52,317 |
| 64-patch local-window occurrences | 33,768 | 43,657 |
| Patch rows per image, median / maximum | 2,268 / 20,412 | 2,464 / 42,840 |
| Largest source image, width × height | 1,773 × 2,271 | 3,529 × 2,372 |
| Largest processed grid expressed in pixels | 1,764 × 2,268 | 3,528 × 2,380 |
| Window count of largest processed image | 336 | 704 |
| Median BF16 local-input payload per request | 221.70 MiB | 192.42 MiB |
| Median BF16 block-7 input activation per request | 241.31 MiB | 209.43 MiB |
| Median BF16 LM-ready embedding per request | 168.92 MiB | 146.60 MiB |

The CPU audit ran the installed runtime's original `get_window_index_thw` method on every distinct recorded grid. All 35 unique grids across the combined cohorts agree exactly with the exported permutation and cumulative local-window bounds. Per-window merge-grid row and column ranges and packed patch spans specify every assigned row without storing original input tensors. [Window and prefix evidence](../mllm_epd_modality_aware_idle_execution_discovery/results/vision_work_geometry.json) · [CPU verifier](../mllm_epd_modality_aware_idle_execution_discovery/vision_geometry.py)

For a geometry-only 50% placement that keeps complete windows and sends packed BF16 patch input to the helper, the largest primary image moves 24,009,216 bytes in and returns 26,132,480 bytes of BF16 activations before block 7. The largest held-out image moves 50,426,880 bytes in and returns 54,886,400 bytes by the same contract. Transporting FP32 processor input doubles the input payload. The full block 7 attention processes the rejoined 20,412 or 42,840 rows, with 416.6 million or 1,835.3 million ordered query–key pairs. These figures quantify a dependency and payload sizes. Synchronization, bandwidth, padding overhead, memory allocation and Decode interference require physical tests.

## Legal image-complete LM prefixes

Every audited visual span was checked against the original cached processor IDs. The audit checks the starting delimiter, every image placeholder and the original closing delimiter. Each reported cut includes the closing delimiter and preserves every earlier token. The first such cut has a median of 607.5 LM rows in the primary and 791 in the held-out, with maxima of 5,119 and 10,726. No single exact closing cut is shared by every request in either cohort. The excess over a target of 4,096 rows at the next valid closing cut has medians of 432.5 and 539 rows. An equal-row comparison should record the chosen image boundary and pair it with a matched arbitrary-token control. [Per-image exact cuts](../mllm_epd_modality_aware_idle_execution_discovery/results/vision_work_geometry.json)

An image-complete prefix is causally legal when its images, full original positions, and any required past KV are ready. A physical scheduling test must establish whether an encoder publishes that prefix early enough to execute useful work.

## Reproduction

```bash
ns=poc_flashvep/mllm_epd_modality_aware_idle_execution_discovery
module=poc_flashvep.mllm_epd_modality_aware_idle_execution_discovery
previous=../h200_epd_modality_runs
dataset=../h200_idle_execution_runs/dataset
python -m "$module.cohorts" --prior-root "$previous" --out "$dataset" \
  --receipts "$ns/results/cohort_receipts.json"

CUDA_VISIBLE_DEVICES='' ../mllm_epd_runtime_env/bin/python -m "$module.vision_geometry" \
  --model-config "$MODEL_CONFIG" --runtime-vision-source "$RUNTIME_VISION_SOURCE" \
  --manifest "$dataset/primary_very_long_n24.json" \
  --manifest "$dataset/heldout_long_short_natural_n34.json" \
  --out "$ns/results/vision_work_geometry.json"
```

The exact model config and installed vision source SHA-256s are recorded in the window artifact. The source checkpoint directory is recorded in the parent campaign's `nixl-ready/state/model_download_complete.json`. The installed vision source is `vllm/model_executor/models/qwen2_5_vl.py` under the parent runtime environment.

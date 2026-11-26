# Discovery200 measured Native EPD baseline

NativeStrong-Free-BI0 completed 200/200 Discovery200 requests in each of two finite-cohort starts. The first used the 0.75 QPS automatic fallback and passed joint SLO on all 200. The complementary grouped-v2 start explicitly used QPS 2.50 and passed joint SLO on 198/200. TTFT p95 was 2,040.3 ms and 2,359.9 ms, respectively. [Original Native receipt](../mllm_specific_disaggregation_discovery/results/native/native_baseline_receipt.json) · [Grouped Native receipt](../mllm_specific_disaggregation_discovery/results/native_grouped/native_baseline_receipt.json)

## Execution contract and load selection

The run used the pinned Qwen2.5-VL-7B-Instruct BF16 model, original vision encoder, original MRoPE and all 28 LLM layers. A single H200 group assigned GPU4 to Encode, GPU5 to Prefill and GPU6 to Decode. The runner uses CUDA IPC for embeddings and Native NIXL for KV handoff. P/D use B8192/S1024, Block128 and KV42 GiB. The prefill and decode workers share the unrestricted baseline configuration, stock compiler and BI0. Prefix caching is disabled. Runtime uses vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129, Transformers 5.16.1 and NIXL 1.4.1. GPU UUIDs and wheel provenance are in the receipt. [Runner](../mllm_specific_disaggregation_discovery/native_epd.py) · [Runtime and Native receipt](../mllm_specific_disaggregation_discovery/results/native/native_baseline_receipt.json)

All cohorts used original prompts, Greedy and Natural EOS, with 2,048 output tokens reserved. Arrivals were open-loop Poisson, request order seed 2026112701 and arrival seed 2026112702. Joint SLO required TTFT ≤5,000 ms, all client ITLs ≤34 ms and request TPOT ≤33 ms/token. Goodput counts output tokens from requests satisfying all three divided by time from first scheduled arrival through drain.

| Cohort | Offered QPS | Completed | Completion req/s | Completion / offered | TTFT p50 / p95 ms | ITL p99 ms | TPOT p95 ms | Decode / joint SLO | Output tokens | Goodput tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 screen 1 | 0.75 | 42/42 | 0.584 | 77.84% | 384.2 / 2,544.8 | 5.86 | 4.97 | 100.00% / 100.00% | 8,847 | 122.97 |
| v1 screen 2 | 2.50 | 42/42 | 1.722 | 68.88% | 875.5 / 2,575.4 | 9.97 | 5.11 | 100.00% / 100.00% | 6,942 | 284.62 |
| v1 screen 3 | 5.00 | 42/42 | 2.537 | 50.74% | 1,959.5 / 3,053.0 | 11.98 | 5.52 | 88.10% / 88.10% | 7,495 | 306.01 |
| Full Discovery200 | 0.75 | 200/200 | 0.708 | 94.39% | 413.3 / 2,040.3 | 6.29 | 4.91 | 100.00% / 100.00% | 31,835 | 112.69 |

The selection rule required at least 80% historical TTFT pass and at least 80% offered-rate completion. All three v1 screens passed the TTFT threshold. All missed the completion-rate dimension in the short finite cohort. The preregistered fallback selected 0.75 QPS. Full-roster throughput at that load was 4,229.86 input+output tokens/s and 112.69 output tokens/s, using a 282.51 s first-arrival-to-drain interval. The short screens used the same request IDs at each load but actual Natural EOS counts varied. Their completion rate includes finite drain and is evidence for the load-selection rule. [Per-cohort measurements and arrivals](../mllm_specific_disaggregation_discovery/results/native/native_baseline_receipt.json)

The three v1 screens each contained 41 grouped-v2 Discovery requests and one request subsequently isolated in Validation, `TIGER-Lab__Mantis-Eval__xh_35`. These screens supply archival baseline diagnostics. Candidate selection uses grouped-v2 Discovery evidence. The full native load included the identical 200 original questions and is reconciled to grouped v2 by request ID without changing its arrivals, input geometry or outputs. [Grouped freeze receipt](../mllm_specific_disaggregation_discovery/dataset_registry/dataset_freeze_receipt.json) · [Per-request ledger](../mllm_specific_disaggregation_discovery/results/native/native_cohorts_requests.csv)

## Grouped-v2 re-screen and complementary QPS 2.50 baseline

All three re-screens used the same 42 grouped-v2 Discovery requests, and none used any of the Validation50. Each met the 80% TTFT-attainment threshold and fell below 80% of offered completion rate under the finite-drain accounting. The automatic rule continued to choose QPS 0.75. The full grouped-v2 Native arm intentionally added QPS 2.50 using `--full-qps 2.5`, recorded as an explicit complementary load.

| Cohort | Offered QPS | Completed | Completion req/s | Completion / offered | TTFT p50 / p95 ms | ITL p99 ms | TPOT p95 ms | Decode / joint SLO | Output tokens | Goodput tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 screen 1 | 0.75 | 42/42 | 0.584 | 77.81% | 486.5 / 2,583.1 | 5.63 | 6.80 | 88.10% / 88.10% | 8,565 | 83.42 |
| v2 screen 2 | 2.50 | 42/42 | 1.689 | 67.57% | 1,132.4 / 2,826.3 | 10.01 | 7.77 | 100.00% / 100.00% | 6,850 | 275.52 |
| v2 screen 3 | 5.00 | 42/42 | 2.645 | 52.89% | 1,832.4 / 2,841.5 | 12.79 | 7.52 | 90.48% / 90.48% | 6,256 | 225.10 |
| Full v2 Discovery200 | 2.50 | 200/200 | 2.333 | 93.33% | 767.2 / 2,359.9 | 10.26 | 6.28 | 99.00% / 99.00% | 33,883 | 383.35 |

Full v2 QPS 2.50 served 13,964.82 total tokens/s and 395.29 output tokens/s over 85.72 s, with two ITL-only SLO misses. Main100 generated 5,425 tokens (median 25.5), MMMU 83 (2), MathVista 10,469 (188), and MMReason 17,906 (421). MathVista and MMReason each had one output cap hit. The original and grouped full starts had the same input IDs and geometry, with 125/200 exact generated sequences matching across loads. Both runs use Natural EOS, so the realized output work differs. [Grouped cohort and scorer receipts](../mllm_specific_disaggregation_discovery/results/native_grouped/native_baseline_receipt.json) · [Grouped load policy](../mllm_specific_disaggregation_discovery/results/native_grouped/config/frozen_native_load.json)

At QPS 2.50 the host E HTTP p50/p95 was 260.17/880.35 ms, P HTTP 418.17/1,425.00 ms, D submit to first SSE 56.19/101.86 ms, and D HTTP 277.15/4,574.04 ms. Client E2E was 1,475.52/5,012.76 ms. The 175 NVML samples reached 15.33 GiB E, 62.75 GiB P and 62.75 GiB D. These intervals and allocation samples describe a single-worker-epoch observation. [Grouped request timings](../mllm_specific_disaggregation_discovery/results/native_grouped/native_timing_ledger.jsonl.gz)

Local scorer points changed for MathVista to 56.67% (17/30) and MMReason to 3.33% (1/30). The MMMU, Mantis, DocVQA, MP-DocVQA and MMLongBench-Doc points below were unchanged. The extended run offers an additional point estimate under the same local scorer limitations and one start per load. [Grouped scorer coverage](../mllm_specific_disaggregation_discovery/results/native_grouped/native_baseline_receipt.json)


## QPS 0.75 stage and memory observations

| Host or client interval | N | Mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| E HTTP, from submit to completion | 200 | 271.50 | 201.71 | 706.51 |
| E completion to P submit | 200 | 0.13 | 0.10 | 0.35 |
| P HTTP, from submit to P-ready | 200 | 361.21 | 151.56 | 1,233.67 |
| P-ready to D submit | 200 | 0.13 | 0.10 | 0.36 |
| D submit to first SSE bytes | 200 | 46.54 | 42.04 | 82.75 |
| D submit to D HTTP completion | 200 | 795.21 | 252.28 | 3,835.23 |
| Client end-to-end | 200 | 1,450.82 | 1,100.71 | 4,075.47 |

HTTP stages include scheduling, transport and runtime execution. D submit to first SSE includes any initial KV import, first output generation and streaming overhead. Stage percentiles describe separate distributions. A CUPTI overlap or kernel-busy claim requires separate traces.

The 568 NVML samples recorded peaks of 16.69 GiB E, 62.75 GiB P and 62.75 GiB D. These are sampled device-wide allocations. The E CUDA IPC epoch ended with 2,372 saves and releases balanced, zero live images and zero outstanding exports. Its observed source-storage high-water mark was about 0.85 GiB. [NVML and ownership receipt](../mllm_specific_disaggregation_discovery/results/native/native_baseline_receipt.json)

Every full-cohort request published a logical prompt KV block descriptor. The reported block count averaged 45.94, peaked at 239 and totaled 9,188 across 200 requests. The original BF16 model has 28 layers, four KV heads, head dimension 128 and Block128. A full logical block maps to 7 MiB of model cache storage, corresponding to mean 321.58 MiB and maximum 1,673 MiB per request from the descriptor counts. A separate transfer observer is required for physical NIXL bytes, transfer duration, transfer overlap and peak per-request live KV residency. [Exported prompt block counts](../mllm_specific_disaggregation_discovery/results/native/native_cohorts_requests.csv)

## Workload and scoring breakdown

| Benchmark | Requests | Output tokens | Output median | TTFT p50 / p95 ms | Client E2E p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main100 | 100 | 5,524 | 25.5 | 894 / 2,422 | 1,214 / 2,794 |
| MMMU | 40 | 83 | 2 | 243 / 729 | 257 / 742 |
| MathVista | 30 | 10,669 | 188 | 206 / 930 | 1,441 / 6,495 |
| MMReason | 30 | 15,559 | 404 | 192 / 556 | 2,064 / 6,136 |

The two longer-generation cohorts shift work toward D despite shorter inputs. The matched run retains that actual work when reporting host stages and E2E. MMMU uses the official concise reply format and ends much earlier. [Dataset and score profile](mllm_discovery200_dataset_profile.md) · [Figure 1](../mllm_specific_disaggregation_discovery/results/figures/figure1_discovery200_dataset_profile.pdf)

| Task | Native point score | Scored | Metric scope |
| --- | ---: | ---: | --- |
| MMMU | 55.00% | 40/40 | Pinned upstream accuracy with stable per-request seed on fallback |
| Mantis-Eval | 48.00% | 25/25 | Pinned upstream evaluation on unchanged text |
| DocVQA | 0.00% | 13/13 | Format-sensitive ANLS on unchanged model text |
| MP-DocVQA | 4.29% | 25/25 | Format-sensitive ANLS on unchanged model text |
| MathVista | 50.00% | 30/30 | Diagnostic local extraction and upstream normalization |
| MMReason | 0.00% | 30/30 | Diagnostic local normalized exact answer |
| MMLongBench-Doc | 2.70% | 37/37 | Diagnostic upstream RAW scorer with extraction unavailable |

DocVQA's zero is the measured unchanged-response ANLS point score, whose span matching is sensitive to verbose output formatting. MMReason's zero describes a local equality check on extracted final strings. The official MMReason assessment judges equivalent solutions and scores intermediate steps under a separate protocol. The official MathVista and MMLongBench extraction paths did not run. Aggregate scores across the different metric scales would be difficult to interpret. [MMReason assessment](https://arxiv.org/abs/2506.23563) · [Pinned quality implementation](../mllm_specific_disaggregation_discovery/quality.py)

## Publication and reproduction

Local raw data live under `h200_discovery_runs/native_free/`. Curated source-level scoring and response metadata are under `h200_discovery_runs/native_analysis/`. The compact publication has 326 request rows across four cohorts, four byte-identical arrival traces, score coverage and exact hashes of output text and token IDs. It keeps exact client token timestamp lists and coordinator endpoint timestamps in a compressed ledger. [Per-request data](../mllm_specific_disaggregation_discovery/results/native/native_cohorts_requests.csv) · [Token/host timing ledger](../mllm_specific_disaggregation_discovery/results/native/native_timing_ledger.jsonl.gz) · [Full arrival trace](../mllm_specific_disaggregation_discovery/results/native/arrival/discovery200_native_arrival.json) · [Screen 1](../mllm_specific_disaggregation_discovery/results/native/arrival/screen_1_arrival.json) · [Screen 2](../mllm_specific_disaggregation_discovery/results/native/arrival/screen_2_arrival.json) · [Screen 3](../mllm_specific_disaggregation_discovery/results/native/arrival/screen_3_arrival.json)

The recorded effective baseline invocation was:

```bash
../mllm_epd_runtime_env/bin/python -u -m poc_flashvep.mllm_specific_disaggregation_discovery.native_epd \
  --asset-repo ../MLLM-EP-mllm-epd \
  --receipt ../mllm_epd_autosetup/download_ready_1795600931728841878.json \
  --manifest ../h200_discovery_runs/dataset/discovery200.json \
  --run-dir ../h200_discovery_runs/native_free --screen-count 42 --wall-seconds 1800
```

From the research worktree, reproduce the compact artifacts from the completed run as follows.

```bash
python -m poc_flashvep.mllm_specific_disaggregation_discovery.native_publication \
  --run ../h200_discovery_runs/native_free \
  --run-manifest ../h200_discovery_runs/dataset/discovery200.json \
  --grouped-manifest ../h200_discovery_runs/dataset/grouped_v2/discovery200.json \
  --native-analysis ../h200_discovery_runs/native_analysis \
  --output poc_flashvep/mllm_specific_disaggregation_discovery/results/native
```

The grouped-v2 supplement preserves its own [four configs and load policy](../mllm_specific_disaggregation_discovery/results/native_grouped/config), [four exact arrivals](../mllm_specific_disaggregation_discovery/results/native_grouped/arrival), [326 compact request rows](../mllm_specific_disaggregation_discovery/results/native_grouped/native_cohorts_requests.csv), and [token/host timing ledger](../mllm_specific_disaggregation_discovery/results/native_grouped/native_timing_ledger.jsonl.gz). Reproduce it with the same publication command using `native_grouped` as the raw run directory, `native_grouped_analysis` as the analysis path, and the grouped-v2 manifest for both manifest arguments. Write the supplement to `poc_flashvep/mllm_specific_disaggregation_discovery/results/native_grouped`.

# E→D whole-request prefill deflection

## Physical path and correctness

D consumes the original E tensor with GPU CUDA IPC, runs the intact LM prefill with original prompt/MRoPE and then decodes on D. The deferred-admission path permits one D-local prefill at a time. It requires D to have no active HTTP generation when admitting the local request and defers new native arrivals until the local request's first generated token. After that token, both kinds of Decode can run on D. This is a whole-request split with D-local Chunked Prefill.

The matched oracle-free request-count policy is causal at full E readiness. It selects D if at least two P HTTP requests are outstanding and D has no active or reserved request. The cost policy additionally estimates P wait, D-local prefill duration and NIXL impact using real LM tokens and observations independent of candidate outcomes. It protects a predicted gap before the next native P completion. Its predictor is a simple FIFO approximation and actual decoder deadlines still require online validation. The measured isolated low-load D reference used for this predictor was about 23,001 LM input token/s. The associated pinned B8192 P finite reference was 31,865 input token/s.

In the initial low-load Fixed16 N4 check, all 64 generated tokens agreed (4/4 complete sequences). Native consumed 743,780,352 logical CUDA IPC clone bytes on P and 5,996,806,144 connector-reported KV bytes on D. D-local consumed the identical 743,780,352 logical IPC clone bytes on D and recorded zero NIXL transfers. Full prompt and output identities are retained in the paired receipts. Local D submit-to-first-token took 1.022–1.235 s. Absolute Decode SLO passed for all four requests in both arms. This checks real execution, transfer and greedy behavior for four requests. Task quality and full KV bitwise agreement require separate probes.

## Matched on-demand pilot

Qwen2.5-VL-7B, three H200s, B8192/S128, 42 GiB KV, the same unmodified Very-long natural N24 inputs, QPS 1.465 Poisson, Fixed16. Both routes use on-demand E CUDA IPC retrieval. Request goodput divides joint passes by planned first arrival to drain. Rates use finite-cohort wall time.

| Route | Completed N | D-local N | TTFT p95 | TPOT p95 | ITL p99 | Joint passes | Goodput req/s | NIXL bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Matched E→P→D | 24 | 0 | 19.05 s | 3.976 ms | 11.19 ms | 1 | 0.034 | 36,024,877,056 |
| Naive E→D | 24 | 4 | 14.92 s | 7.261 ms | 18.31 ms | 3 | 0.119 | 29,719,789,568 |

The four local requests completed 109,581 LM input tokens. Naive deflection reduced NIXL bytes by 17.50% and matched-pilot TTFT p95 by 21.67%. TPOT and ITL increased by 82.63% and 63.60%, respectively. All 24 requests satisfied the absolute Decode SLO in both arms. Strict Decode isolation failed. The optimized Native EPD on the same earlier roster had 10.99 s TTFT p95, ahead of this on-demand implementation outside the matched pilot. The on-demand implementation can wait for E's export RPC even when the required embedding already exists. This pilot does not establish a gain over NativeStrong.

[Raw tiered output](../mllm_epd_prefill_heavy_modality_discovery/results/whole_request_routes.csv) · [Paired token and arrival receipts](../mllm_epd_prefill_heavy_modality_discovery/results/whole_request_paired_tokens.json)

## NativeStrong inline transfer control

The existing one-shot inline CUDA IPC descriptor can be claimed by the final D consumer for a D-local request. The router resolves that request to exactly one recipient. The native-only request can still use its original inline P descriptor. D observed 189 verified redirected inline image clones and no native NIXL reads in the Fixed16 N4 control. Identical full prompt and output sequences were observed for all four paired requests.

NativeStrong, no-deflection inline router, and both deflection policies ran on the same natural N24 roster and scheduled offsets. The reported engine setting is B8192/S128, KV42 GiB and QPS 1.465. The cost predictor uses 69.63 GB/s as the observed baseline aggregate NIXL bytes divided by native transfer duration, as a shared-fabric approximation. This rate includes the NIXL metric's scope and gives no guarantee about the latency of an individual E→D clone.

| Route | D-local N | TTFT p95 | TPOT p95 | ITL p99 | Decode SLO | Joint passes | Goodput req/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| NativeStrong | 0 | 10.519 s | 4.504 ms | 11.326 ms | 24/24 | 8/24 | 0.384 |
| Matched inline router | 0 | 10.214 s | 3.626 ms | 12.928 ms | 24/24 | 8/24 | 0.384 |
| Naive E→D inline | 5 | 7.486 s | 6.370 ms | 22.883 ms | 24/24 | 15/24 | 0.852 |
| Cost-gated E→D inline | 0 | 11.112 s | 3.755 ms | 11.752 ms | 24/24 | 9/24 | 0.434 |

Against NativeStrong, naive deflection reduced TTFT p95 by 28.8% and increased finite-cohort joint goodput 2.22×. Its TPOT p95 increased 41.4% and its ITL p99 increased 102.0%, outside the 5% strict isolation gate. Actual D-local work was 130,754 prompt tokens across five requests. Native NIXL bytes fell 36.025→28.509 GB. D fetched 933,524,480 logical embedding bytes across 175 original images from E. The conservative cost gate admitted zero in this timing regime.

Paired NativeStrong versus naive retained exact full prompt IDs on 24/24 requests and exact Fixed16 output on 22/24. Both first output differences appeared at step 14, zero-indexed. Of the two differences, one request was D-local and one was routed through original P. Fixed16 agreement and isolated same-prefix logits have limited scope for benchmark quality.

## Matched representation-only increment

At fixed 16K, QPS 2.600, Fixed16 and four mixed visual-ratio bins, both cost selectors used the same calibrated device/queue estimates. The control estimated IPC bytes using the cohort median visual ratio, and the per-request policy used the actual image rows available when E finished. This changes a small transfer term while preserving total LM input length and KV bytes.

| Route | D-local N | TTFT p95 | TPOT p95 | ITL p99 | Joint goodput req/s |
|---|---:|---:|---:|---:|---:|
| NativeStrong | 0 | 5.414 s | 4.04 ms | 12.22 ms | 1.843 |
| Cost with cohort representation estimate | 2 | 4.487 s | 5.98 ms | 16.47 ms | 2.319 |
| Cost with actual per-request representation | 2 | 4.502 s | 7.13 ms | 18.27 ms | 2.322 |

Both selectors completed 24/24 with the same prompt and full output IDs as NativeStrong; neither passed the strict relative Decode protection check. Per-request representation changed which requests were selected. Joint attainment was 24/24 in both cost arms and finite-cohort goodput differed by 0.1% in the pilot. The measured increment did not reach the 15% target.

## Decode budget and Natural EOS

A new paired start fixed the P budget at 8,192 and used a 4,096-token D budget in every arm. The isolated variant keeps the D-local admission lock through the last token. The ordinary naive arm releases the lock at the first generated token. Both wait for the local request's full prefill and both use the inline redirect.

| Fixed16 natural N24, QPS 1.465 | D-local N | TTFT p95 | TPOT p95 | ITL p99 | Absolute Decode | Joint goodput req/s |
|---|---:|---:|---:|---:|---:|---:|
| NativeStrong | 0 | 10.602 s | 4.331 ms | 11.078 ms | 24/24 | 0.431 |
| Matched no-deflection | 0 | 10.585 s | 3.892 ms | 8.398 ms | 24/24 | 0.430 |
| Naive inline | 6 | 7.928 s | 7.147 ms | 22.846 ms | 21/24 | 0.685 |
| Cost-gated inline | 0 | 10.413 s | 4.063 ms | 10.751 ms | 24/24 | 0.384 |
| Naive inline with full-request D isolation | 3 | 8.515 s | 5.725 ms | 17.728 ms | 23/24 | 0.425 |

A smaller D token budget and a longer admission lock changed service timing and the selected work. Neither protected the 5% relative tails. The absolute gate evaluates every ITL and catches request failures even when aggregate ITL p99 is below 34ms.

| Natural EOS, Main400 mix N64, QPS 2.955 | D-local N | TTFT p95 | TPOT p95 | ITL p99 | Joint passes | Joint goodput req/s |
|---|---:|---:|---:|---:|---:|---:|
| NativeStrong | 0 | 3.557 s | 5.119 ms | 14.247 ms | 63/64 | 2.870 |
| Naive inline | 6 | 3.613 s | 6.737 ms | 14.741 ms | 62/64 | 2.814 |

Both completed 64/64. Exact full prompt IDs agreed on 64/64. Entire generated token sequences agreed on 53/64 overall and on 4/6 D-local requests. The existing EPD is a stronger service-level result on this Natural EOS mix. [Original task scoring](mllm_epd_natural_eos_task_quality.md)

## Same-prefix full-vocabulary logits

Three serialized EPD A/A/B comparisons retain a single original GPU E embedding export under verified ownership. A runs native E→P→D and generates a 15-token teacher prefix. A′ and B receive that same teacher prefix, with B running full E→D local prefill. Prompt IDs, actual last input token, and actual MRoPE positions agreed at all 48 evaluated steps. Every step compares all 152,064 vocabulary logits before teacher forcing.

| Original request | Native A/A bitwise | Native/D greedy | Native/D bitwise | Max absolute logit gap | Max relative L2 |
|---|---:|---:|---:|---:|---:|
| DocVQA, 1,341 input | 16/16 | 16/16 | 0/16 | 0.2500 | 0.01866 |
| Mantis, 1,446 input | 16/16 | 15/16 | 0/16 | 0.7031 | 0.06039 |
| Very-long MMLongBench-Doc, 24,761 input | 16/16 | 15/16 | 0/16 | 0.3750 | 0.05769 |

The Mantis mismatch was step 9, zero-indexed. The Very-long mismatch was step 14, the same original request and step as the D-local divergence in the online B8192 N24. Its measured top-two margin changed from 0.0625 under native to a tie on D-local. The first emitted token uses one Native D query row in the FULL graph. D-local uses 1,341, 1,446 or 486 rows of the final prefill invocation outside the decode graph, depending on the input. The tested implementation preserves shared source tensors and token/position semantics but exceeds the observed A/A numerical variation. Layer-by-layer KV reconstruction and the cause of the observed logit gaps remain unverified.

[C0–C1 compact per-step receipts](../mllm_epd_prefill_heavy_modality_discovery/results/teacher_logits/) · [Natural-EOS task scores](mllm_epd_natural_eos_task_quality.md)

## Independent B8192 Fixed16 replication

The later B8192 epoch uses the same natural Very-long 24-source roster cycled to N64, separate media occurrence identities, QPS 1.465, exact seeded arrivals, and a full-cohort prime before measured routing. All 64 requests completed in each arm.

| Route | Local requests / prompt rows | TTFT p95 | TPOT p95 | ITL p99 | Absolute Decode | Joint passes | Goodput req/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| NativeStrong | 0 / 0 | 14.717 s | 4.162 ms | 12.330 ms | 64/64 | 7/64 | 0.131 |
| Naive inline | 11 / 296,639 | 8.883 s | 6.917 ms | 21.541 ms | 63/64 | 16/64 | 0.354 |
| Naive inline with full-request D isolation | 9 / 235,028 | 11.703 s | 4.689 ms | 12.391 ms | 64/64 | 10/64 | 0.210 |

Isolation retained the absolute Decode SLO and the ITL p99 relative threshold. It increased TPOT p95 by 12.7%. Exact Fixed16 outputs versus native were 61/64 for naive and 63/64 for isolation, with D-local agreement 10/11 and 8/9. NIXL bytes were 96.485/79.434/82.972 GB. The replication establishes a bounded Fixed16 service-level trade-off on this workload. It leaves strict quality, absolute capacity at steady state, and performance on other MLLM architectures open.

## Same-budget colocated DP3

The same three H200s each run an independent BF16 model including vision, plain request round-robin, APC off and 42 GiB KV per replica. All prompts in the aligned arms matched the NativeStrong prompt token IDs. Each replica had an original-image warmup. The subsequent Very-long N64 followed the complete N24 workload on those replicas. The baseline uses no E→P or P→D handoff.

| B8192 DP3 workload | TTFT p95 | TPOT p95 | ITL p99 | Absolute Decode and joint | Joint goodput req/s |
|---|---:|---:|---:|---:|---:|
| Very-long N24, QPS 1.465, Fixed16 | 3.762 s | 306.936 ms | 538.294 ms | 4/24 | 0.291 |
| Very-long N64, QPS 1.465, Fixed16 | 3.531 s | 331.666 ms | 484.737 ms | 22/64 | 0.535 |
| Mixed visual 16K N24, QPS 2.600, Fixed16 | 2.077 s | 156.516 ms | 392.552 ms | 12/24 | 1.607 |
| Main400 mix N64, QPS 2.955, Natural EOS | 2.497 s | 51.013 ms | 361.911 ms | 38/64 | 1.747 |

At Very-long N64 this DP3 setting's joint goodput is higher than the naive EPD result of 0.354 req/s and its Decode SLO pass count is 22/64. The naive EPD pass count is 63/64 with ITL p99 21.54ms. The comparison therefore yields a different TTFT/Decode trade-off under the same GPU budget. For the tested Natural EOS mix both EPD arms delivered higher joint goodput and more Decode SLO passes than this DP3 setting.

The lower-budget DP3 control repeats the same four arms with a 2,048-token batch budget to reduce colocated prefill stalls.

| B2048 DP3 workload | TTFT p95 | TPOT p95 | ITL p99 | Absolute Decode and joint | Joint goodput req/s |
|---|---:|---:|---:|---:|---:|
| Very-long N24, QPS 1.465, Fixed16 | 3.721 s | 111.083 ms | 173.688 ms | 7/24 | 0.499 |
| Very-long N64, QPS 1.465, Fixed16 | 3.628 s | 110.005 ms | 150.751 ms | 28/64 | 0.680 |
| Mixed visual 16K N24, QPS 2.600, Fixed16 | 2.054 s | 98.435 ms | 140.919 ms | 12/24 | 1.579 |
| Main400 mix N64, QPS 2.955, Natural EOS | 2.335 s | 58.167 ms | 120.345 ms | 39/64 | 1.793 |

The tighter budget brought down the observed Fixed16 tail and raised long-N64 joint goodput. The strongest DP3 tested joint rate for that setting exceeds the EPD deflection rate. EPD deflection delivers many more requests under the absolute Decode SLO. The Natural EOS gain remains with Native EPD. DP3 uses three local model instances and three KV reservations, which is feasible on the same GPUs. Round-robin queue assignment and the two measured batch budgets define the tested baseline range.

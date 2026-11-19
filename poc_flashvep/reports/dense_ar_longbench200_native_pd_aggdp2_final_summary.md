# LongBench v1 Mixed200: Native PD 재현과 Agg DP2 비교

## 결론

LongBench v1의 실제 요청200개를 동결했다. Native PD와 Agg DP2의 Main22회에서4400개 요청의 Output Token Sequence가 같은 Reference와 일치했다. 요청한 TTFT p95, ITL p99, TPOT p95, Output Throughput Figure4개를 PNG/PDF로 저장했다.

기존22–25ms Decode Step은 공식 vLLM의 Batch-invariant 경로에서도 재현됐다. 같은 모델의 기본 BI0 경로는 약7–8ms였다. 따라서 기존 지연을 Custom PD만의 문제로 설명할 수 없다. 엄격한 Output 일치 계약이 선택한 Numerical 경로의 성능 비용을 확인했다.

높은 부하에서 P Compute 병목은 확인했다. 그러나 D의 안전한 Offloading Headroom은 확인하지 못했다. D Kernel Gap의 대부분은 NIXL 전송 API 구간과 겹쳤다. Native의 Decode Tail도 Frozen SLO를 넘었다. Layer/Token과 Dynamic 실험은 재개하지 않았다.

| 독립 판정 | 결과 |
|---|---|
| LongBench v1 Dataset200 동결 | PASS |
| Model/Output/State 정합성 | PASS_IN_OFFICIAL_BATCH_INVARIANT_FAMILY |
| 기존22–25ms Model Decode 재현 | EXPLAINED_BY_NUMERICAL_PATH_CONTROL |
| Concurrent NIXL Tail 원인 분해 | PARTIAL_ROOT_CAUSE_IDENTIFICATION |
| Native End-to-end 재현성 전체 기준 | PARTIAL / HOLD |
| 높은 부하의 P Compute 병목 | CONFIRMED |
| Decode-safe Headroom과 Motivation 전체 | NOT_ESTABLISHED / HOLD |
| Layer/Token 및 Dynamic 신규 성능 | NOT_RUN |
| 장기 Sustainable Capacity | NOT_MEASURED |

## 1. 공통 실행 계약

| 항목 | 실제 설정 |
|---|---|
| 모델 | Qwen/Qwen3-8B |
| Model Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| 계산 | Original BF16, Exact Causal Attention, Full36 Layers |
| 생성 | Greedy, Thinking Disabled, Natural EOS, 공통 Output Cap2048 |
| Runtime | vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1 |
| GPU | H100 SXM80GB×2, Primary0/1 |
| GPU0 UUID | `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d` |
| GPU1 UUID | `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb` |
| Worker 설정 | 명시적 B8192/S1024, KV42GiB/Engine, Prefix Cache 비활성화 |
| Attention / Numerical 경로 | FLASH_ATTN/HND, 공식 Batch-invariant 모드 |
| Agg | 독립 Full-model Replica2개, Round-robin, Standard Continuous Batching |
| Native | GPU0 Full Prefill, GPU1 Decode, 공식 NIXL Connector, Standard Scheduler |
| Compile / Scheduler | Compile3, 기본 Graph 정책, Async Scheduling |
| Frozen SLO | TTFT≤5000ms, 모든 ITL≤34ms, TPOT≤33ms |

Native는 공식 Toy Proxy와 같은 Prompt 계약을 사용했다. P의 첫 Sample은 버렸다. D는 원래 Prompt의 마지막 Query1개를 다시 계산했다. P의 전체 Prefill을 D에서 재계산하지 않았다. [공식 NIXL 예제](https://github.com/vllm-project/vllm/blob/v0.26.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py).

이번 실행은 설치된 `LLM` API, Local Worker와 Coordinator를 사용했다. HTTP와 Detokenization 비용은 포함하지 않는다. Main의 수치는 Stock BI0 Production 성능이 아니다. 두 방법의 엄격한 Greedy Output 일치를 확보한 BI1 Family의 비교다.

기존 사용자 Checkout, Staged Index와 외부 Runtime은 변경하지 않는다. 새 Source와 Raw Evidence는 별도 Namespace에 저장한다. 실행별 Source ZIP SHA256, Command, Worker Configuration과 GPU Map을 보존한다.

## 2. Dataset200과 실제 생성 길이

LongBench v1 Revision은 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`다. HotpotQA, MuSiQue, NarrativeQA, GovReport, QMSum을 각각40개 추출했다. Sampling Seed는2026111999와 Task Index0–4를 사용했다. Mixed Order Seed는2026112000이다.

Dataset SHA256은 `1a9e6637bbd4f26e1338533d6735fa981c3ab258e0a59ac39c35b8b22d035553`이다. 원문 ID, Context, Prompt, 최종 Token ID와 개별 Hash를 보존했다. 기존 Pool과 겹치는 요청은18개다. 전체200개를 완전히 새로운 Held-out Dataset이라고 부르지 않는다.

| 실제 길이 | Input Tokens | Natural Output Tokens |
|---|---:|---:|
| 평균 | 16674.06 | 129.325 |
| p50 | 16021 | 12 |
| p90 | 33320.9 | 528 |
| p95 | 38912 | 605.1 |
| p99 | 38912 | 651.18 |
| 최소 / 최대 | 1910 / 38912 | 2 / 717 |
| 전체 Token 수 | 3334812 | 25865 |

Input은 Qwen Chat Template 이후 길이다. Context Limit40960과 Output Reserve2048을 적용했다.11개는 Head/Tail 방식으로 Input38912 Tokens까지 줄였다. 제외한 요청은0개다. Truncation은 성능 결과 확인 전에 동결했다. Output Cap에 도달한 요청도0개다. EOS Token은 출력 수에 포함한다.

QA120개는 Input Token 수의70.17%를 차지했다. Summarization80개는 Output Token 수의96.40%를 차지했다. 이는 Token 수의 `STRUCTURAL_ACCOUNTING`이다. GPU Compute Time의 비중은 아니다. [Task별 길이와 전체 계약](dense_ar_longbench200_dataset_contract.md)을 참조한다.

## 3. 기존 Decode 지연의 원인

PD, NIXL과 프로젝트 Model Adapter를 사용하지 않는 Standard Single-GPU 진단을 실행했다. Original BF16과 동일한 Prompt를 유지했다. Flag-only 비교의 Mean TBT는 다음과 같다.

| Input / Output | 공식 BI0 | 공식 BI1 | 독립 Start |
|---|---:|---:|---:|
| 4096 / 16 | 7.1205ms | 20.9115ms | Flag당1회 |
| 8192 / 16 | 7.3243ms | 22.2404ms | Flag당1회 |
| 16384 / 16 | 7.6779ms | 24.6510ms | Flag당1회 |

각 길이는 Warmup2개와 측정6개를 사용했다. Compile0, Full Decode Graph와 Async Scheduler를 사용한 진단이다. Output Token 일치는 BI0/BI1 사이에서 길이별4/6,6/6,5/6이었다. Numerical 경로가 다른 결과를 동일한 정확도 조건의 성능으로 합치지 않는다. [공식 Batch-invariance 문서](https://docs.vllm.ai/en/v0.26.0/features/batch_invariance/)도 성능 비용을 설명한다.

별도의 BI1 Sync Control에서는 Step당 GPU Kernel Union23.5624ms를 측정했다. Engine Step Wall은27.5747ms였다. Inter-step Host Gap은0.1466ms였다. 계측의 Mean TBT 변화는4.9747%로5% Screen을 간신히 통과했다. 이 Sync 수치를 Async22ms에서 빼지 않는다. BI0 Profile은18.52% 교란으로 `DIAGNOSTIC`만 사용한다.

Concurrent Native의 긴 Tail은 별도 문제다. NIXL API의 긴 Host Enclosure와 작은 호출 Thread CPU Time을 관측했다. SDK 내부 Thread Pool의 대기를 Source에서 확인했다. 그러나 실제 DMA, 내부 대기와 GPU 의존성의 완전한 인과 분해는 `NOT_IDENTIFIED`다. [Decode 원인 분석](dense_ar_native_decode_latency_root_cause.md)에 Raw Evidence와 실패한 가설을 보존했다.

Clean C1은 Standard/Native 각각 독립 Start3회를 완료했다. Mean TBT는23.192/22.697ms였다. Native/Standard ITL p99 비율은0.96785[CI95:0.95846,0.97676]이었다. TPOT p95 비율은0.96795[0.95602,0.98105]이었다. 두 상한은1.05 이하였다. Unpaired Start-level Bootstrap729개 조합이며, 초기 Start의 Observer Source 차이를 포함한 진단이다. Concurrent Import와 Mixed200의 Decode 보호를 대신하지 않는다.

별도 N8의 UCX 로그는 CUDA IPC zero-copy 선택을 보고했다. Main의 실제 DMA Time이나 Bandwidth를 측정했다는 뜻은 아니다. Main Native11회에서는 완료 후4.456–6.681초에 SDK Context 종료 오류를 관측했다. Output, KV Credit과 Return Code0은 통과했다. SDK 종료 순서는 미해결 제한으로 남겼다. 이 제한까지 포함해 End-to-end Reproduction은 `PARTIAL / HOLD`다.

## 4. Main QPS 결과

N40 Pilot 이후 공통 QPS0.4/0.7/1/1.15/1.2/1.65/2.35를 동결했다. 길이가 다른 LongBench Dataset에 이전2–10QPS를 그대로 강제하지 않았다. QPS1과1.65는 Method당 독립 Start3회다. 나머지는1회다. 두 방법은 동일한 Prompt, 요청 순서, Arrival Trace와 Natural Output을 사용했다.

다음 값은 Start별 통계의 평균이다. 단위는 TTFT초, ITL/TPOT밀리초, Throughput출력Tokens/s다.

| Offered QPS | Method | Starts | TTFT p95 s | ITL p99 ms | TPOT p95 ms/token | Output tokens/s | Joint SLO % |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.4 | Native | 1 | 2.458 | 49.47 | 32.67 | 49.76 | 63.00 |
| 0.4 | Agg | 1 | 1.928 | 91.83 | 34.44 | 49.79 | 69.50 |
| 0.7 | Native | 1 | 2.765 | 99.74 | 34.20 | 85.45 | 53.50 |
| 0.7 | Agg | 1 | 1.938 | 320.88 | 72.50 | 85.48 | 60.00 |
| 1.0 | Native | 3 | 4.229 | 131.42 | 36.56 | 128.70 | 45.83 |
| 1.0 | Agg | 3 | 1.984 | 337.15 | 176.97 | 129.02 | 43.50 |
| 1.15 | Native | 1 | 4.007 | 136.10 | 36.78 | 136.06 | 43.50 |
| 1.15 | Agg | 1 | 1.973 | 335.91 | 119.10 | 136.31 | 46.50 |
| 1.2 | Native | 1 | 4.612 | 132.13 | 38.26 | 141.50 | 41.50 |
| 1.2 | Agg | 1 | 1.983 | 343.63 | 177.17 | 141.77 | 44.00 |
| 1.65 | Native | 3 | 20.391 | 149.02 | 39.07 | 175.93 | 3.33 |
| 1.65 | Agg | 3 | 2.508 | 416.88 | 298.67 | 202.24 | 23.33 |
| 2.35 | Native | 1 | 46.554 | 138.62 | 38.89 | 176.83 | 3.00 |
| 2.35 | Agg | 1 | 5.067 | 460.78 | 346.51 | 250.99 | 8.50 |

QPS1.65의 Native/Agg TTFT p95 비율은8.1316[CI95:6.1875,9.7377]이었다. ITL p99 비율은0.3575[0.3500,0.3639]이었다. TPOT p95 비율은0.1308[0.1162,0.1501]이었다. Raw Output 비율은0.8699[0.8220,0.9187]이었다. 같은 Seed Pair를 재표본화한 Start-level Bootstrap27개 조합이다.

Native는 Agg보다 Decode Tail을 줄였지만, 절대 Decode SLO를 만족하지는 못했다. Agg는 더 낮은 TTFT와 더 높은 고부하 Raw Throughput을 제공했다. QPS1.65의 SLO-qualified Output은 Native0.281tokens/s, Agg2.748tokens/s였다. Native가 전체 SLO-goodput에서 우세하다는 근거는 없다.

모든 요청을 Drain했다. 수치는 Finite-cohort Throughput이며 Sustainable Capacity가 아니다. Offered, 실제 Arrival, Dispatch와 Completed QPS를 별도로 기록했다. [전체 Table와 Start-level95% CI](dense_ar_longbench200_qps_sweep_results.md)를 참조한다.

## 5. 실제 P/D 상태와 Headroom

동일한 Native Profile 구간에서 QPS1의 P/D Kernel Busy는69.13%/89.18%였다. 이 구간은 P-prefill-heavy가 아니었다. QPS2.35는99.34%/85.22%였다. 같은 구간의 P Waiting 평균은36.77개였다. D Active Decode 평균은5.98개였다. Decode-ready Waiting 평균은0이었다.

QPS2.35의20ms 이상 D Kernel Gap 시간 중97.22%는 NIXL 전송 Host 구간과 겹쳤다. 이 관계는 원인 제거 실험이 아닌 `TRACE_ANALYSIS`다. Kernel Inactivity14.78%를 안전 실행 Budget으로 부르지 않는다. Profile1회/점이며 약86초 ROI다. Observer Screen의 최대 변화는1.56%로5% 이내였다.

따라서 P 병목은 확인했지만, Decode SLO를 보호하는 D Safe Window는 `NOT_ESTABLISHED`다. [Prefill-heavy 검증](dense_ar_longbench200_prefill_heavy_feasibility.md)에 Queue, Kernel Gap, 계측 범위와 이전 실행 Family의 차이를 기록했다.

Main의 Host Timeline에서도 P 대기가 지배적이었다. QPS2.35의 Mean TTFT26904.91ms 중 Arrival→P 첫 Prepare는25299.17ms였다. P 첫 Prepare→Ready는1313.26ms였다. Ready→D Prepare는229.34ms, D Prepare→첫 Delivery는63.14ms였다. 이 네 평균 구간은 같은 요청의 TTFT를 분해한다. GPU-exclusive Time이나 p95 분해로 해석하지 않는다.

## 6. Figure4개

Figure는 Repository의 `poc_flashvep` 바로 아래에 저장했다. 모든 Figure는 동일한 QPS Grid와 두 Method를 사용한다. QPS1/1.65에는 Start3회의95% CI를 표시했다. 나머지 점에는 CI를 만들지 않았다.

1. [TTFT p95 PNG](../longbench200_ttft_p95_vs_qps.png) / [PDF](../longbench200_ttft_p95_vs_qps.pdf).
2. [Client-visible ITL p99 PNG](../longbench200_itl_p99_vs_qps.png) / [PDF](../longbench200_itl_p99_vs_qps.pdf).
3. [Request TPOT p95 PNG](../longbench200_tpot_p95_vs_qps.png) / [PDF](../longbench200_tpot_p95_vs_qps.pdf).
4. [Generated Output Tokens/s PNG](../longbench200_output_tokens_s_vs_qps.png) / [PDF](../longbench200_output_tokens_s_vs_qps.pdf).

Mean TBT, Completed Requests/s, Task별 성능과 SLO-qualified Throughput도 Raw Table에 포함했다. Kernel Busy와 Scheduler Queue는 별도 보조 Figure다.

이 결과는 DuetServe-style Methodology를 새 Dataset에 적용한 Benchmark다. 원 논문의 Input8000/Output200, vLLM0.10.1과 수치 재현이라고 부르지 않는다. [DuetServe 원문](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)의 Figure2는 DP2 Round-robin이다.

## 7. 설정과 한계

실제44개 Main Worker에서 B8192/S1024를 확인했다. 실제 GPU Batch는 별도 기록했다. 예를 들어 QPS1.65의 첫 Native Start에서 P Batch는1–3개였다. D Batch는1–11개였다. Agg도 두 Replica에서 Prefill/Decode Mixed Iteration을 수행했다. 고정 Cohort나 Batch1을 Main에 강제하지 않았다.

QPS0.4의 Native/Agg Wrapper는 Allocation Observer 추가 시점 때문에 Byte가 완전히 같지 않았다. Model Math와 설정은 같았다. 다른 QPS의 Pair는 Runtime 파일이 일치했다. 이 제한은 [Source 감사](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_full_analysis.json)에 보존했다.

B/S Sensitivity는 N40의 Low/Medium/High Load에서 한 변수씩 변경했다.24개 Cell이 모두 Output과 Lifetime을 통과했다. QPS2.35 Agg의 ITL p99는 B4096/8192/16384에서217.42/393.40/618.29ms였다. 각 Cell은 Start1회다. Low-load에서는 같은 단조 관계가 없었다. 이를 전체 Best-tuned Frontier나 Held-out 성능으로 부르지 않는다. Primary 설정은 Main 이전에 동결했다. [전체 Sensitivity와 한계](dense_ar_longbench200_qps_sweep_results.md)에 별도로 기록했다.

Moderate-load Native B8192/S1024 Control은 확장 Queue Observer 추가 전에 실행됐다. 나머지 Moderate Control에는 해당 Read-only Observer가 있다. Model Math는 같지만 Wrapper Source는 완전히 같지 않다. 작은 B/S 차이를 확정적인 인과 효과로 해석하지 않는다. 실행별 Runtime Hash를 별도로 기록한다.

설치된 vLLM Python Source2234개의 RECORD Hash는 일치했다. 이 검사만으로 모든 Binary와 Dependency의 무변경을 주장하지 않는다. GPU-exclusive Kernel Time, CUDA Enclosure, Engine Wall, Host Gap과 Client-visible ITL은 분리했다. Online Kernel별 독점 비용의 완전한 분해와 DMA-only Time은 `NOT_MEASURED`다.

## 8. 다음 연구의 Gate

Native Model 경로는 독립 Reference와 비교할 수 있게 됐다. 실제 Dataset과 공통 QPS 결과도 확보했다. 그러나 Concurrent NIXL Tail과 Deadline 기반 Safe Window가 남아 있다. 이 문제를 해결하기 전에는 Layer/Token의 순이익이나 Dynamic Scheduling 성공을 주장하지 않는다.

기존 Layer/Token 실행 중단과 로그 보존은 [Safe-stop 보고서](dense_ar_previous_experiment_safe_stop.md)에 기록했다. 중단 과정에서 기존 Layer Child는 완료됐다. 기존 Controller가 예기치 않게 시작한 Token Child는 중단했다. 두 기록을 보존했다. Safe-stop 이후 Layer/Token 실험은 재개하지 않았다.

새 Native 캠페인의 실패도 보존했다. 초기 P-token 추가 경로는 NIXL Block 수 Assertion으로 실패했다. 최초 Standard 진단은 `ninja` 실행 경로 문제로 실패했다. 별도 BI0 Clean Control1개는 미완료 상태로 남겼다. Numerical 불일치, 실행 실패와 Observer Screen 실패는 서로 다른 범주로 기록한다. [실행 Inventory](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/run_and_failure_inventory.json)를 참조한다.

## 9. 재현성과 Publication

Raw Namespace는 `dense_ar_native_pd_real_workload_reproduction_20261119`다. 새 Branch는 `flashvep/dense-ar-native-pd-real-workload-reproduction-20261119`다. 사용자 Checkout과 Index를 변경하지 않는 Private-index Publication을 사용한다. 큰 Raw 파일은 원본 SHA256을 검증하는 Lossless Archive로 보존한다.

예정한 GPU 검증은 모두 끝났다. 새 캠페인의 실행 Inventory에는 완료90개, 실패2개, 보존한 미완료1개가 있다.90개는 진단과 Reference도 포함한다. Main은22개 Start다. 새로운 Layer/Token 측정은 추가하지 않았다.

GPU0–7의 승인된 Burner를 복원했다. 소유권과 실제 GPU 사용을5초 간격3회 확인했다. 사용자 Checkout, Index와 외부 Runtime의 보존 검사도 통과했다.10개 CPU Test와 Runtime Freeze9개 파일 검사도 통과했다. 근거는 `state/CAMPAIGN_CLOSED.json`에 저장했다.

GitHub Push와 Remote SHA 확인은 Resource Closeout 뒤에 수행한다. 완료 Receipt는 `state/publication_receipt.json`에 저장한다. Push 이후에도 Burner와 Remote SHA를 재확인한다. 최종 Commit SHA는 전달 메시지와 Local Receipt에 기록한다. Commit 안에 자기 자신의 SHA를 넣지는 않는다.

큰 Raw 원본은 Lossless Archive를 복원한 뒤 GitHub Checkout에서 확인할 수 있다. 서버의 원본 파일은 삭제하지 않는다. Archive Manifest는 각 원본의 SHA256과 복원 방법을 제공한다. 실패 로그와 종료 오류도 동일하게 보존한다.

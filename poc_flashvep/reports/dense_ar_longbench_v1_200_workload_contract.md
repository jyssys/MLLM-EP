# LongBench v1 200-Request 사전 Workload 계약

상태: `INPUT_SELECTION_FROZEN / OUTPUT_NOT_GENERATED / GPU_CONTEXT_GATE_PENDING`.

이 문서는 새 200개 요청을 GPU Serving에 제출하기 전에 작성했다. Natural Output 분포는 아직 측정하지 않았다. 정답 Text의 길이를 생성 Output Length로 사용하지 않는다.

## Source와 선택

- 공식 Dataset: [zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench).
- Dataset Revision: `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`.
- `data.zip` SHA256: `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`.
- 공식 HF API에서 Revision과 LFS SHA256을 확인했다. 같은 Blob의 기존 `THUDM/LongBench` Cache를 재사용했다.
- 공식 Prompt Template Commit: `2e00731f8d0bff23dc4325161044d0ed8af94c1e`.
- Sampling Seed: `2026111999`. Task 순서에 따라 Seed에 0–4를 더했다.
- Mixed Request Order Seed: `2026112000`.

각 Task의 원본 200개 중 40개를 균일하게 비복원 추출했다. Token Length와 모델 Output을 보기 전에 Source ID와 순서를 동결했다.

| Task | Category | Requests |
|---|---|---:|
| HotpotQA | Multi-document QA | 40 |
| MuSiQue | Multi-document QA | 40 |
| NarrativeQA | Long-document QA | 40 |
| GovReport | Long-document Summarization | 40 |
| QMSum | Meeting Summarization | 40 |
| 합계 | QA120 / Summarization80 | 200 |

서로 다른 Request ID는 200개다. 서로 다른 Context SHA256은 163개다. 같은 문서의 다른 질문을 독립 문서로 집계하지 않는다.

직전 144개 문서 Pool과 Source ID가 겹치는 요청은 18개다. 이 요청을 제거하거나 다른 요청으로 교체하지 않았다. 따라서 모든 문서가 과거 연구에서 전혀 사용되지 않았다는 주장은 하지 않는다.

## Tokenizer와 입력

모델은 `Qwen/Qwen3-8B`다. Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.

공식 Task Prompt Template에 원문 Context와 질문을 넣었다. 그다음 Qwen Chat Template를 적용했다. `enable_thinking=False`, `add_generation_prompt=True`를 사용했다.

다음 길이는 실제 Qwen Token 수다. 공식 LongBench의 영어 단어 수 기반 `Avg len`과 다르다. [공식 Dataset 설명](https://huggingface.co/datasets/zai-org/LongBench)도 이 차이를 명시한다.

| 통계 | 원본 Chat Input Tokens | Context 처리 후 Input Tokens |
|---|---:|---:|
| Mean | 17,758.97 | 16,674.06 |
| Min | 1,910 | 1,910 |
| p10 | 6,804.30 | 6,804.30 |
| p25 | 10,097.75 | 10,097.75 |
| p50 | 16,021.00 | 16,021.00 |
| p75 | 17,470.25 | 17,470.25 |
| p90 | 33,320.90 | 33,320.90 |
| p95 | 41,908.45 | 38,912.00 |
| p99 | 65,401.04 | 38,912.00 |
| Max | 65,409 | 38,912 |

| Task | 실제 Input Mean | 실제 Input p50 | 실제 Input Max | Truncated |
|---|---:|---:|---:|---:|
| HotpotQA | 13,366.98 | 14,689.0 | 17,305 | 0/40 |
| MuSiQue | 16,176.48 | 16,794.5 | 17,894 | 0/40 |
| NarrativeQA | 28,960.78 | 32,979.5 | 38,912 | 10/40 |
| GovReport | 10,607.68 | 8,838.0 | 38,912 | 1/40 |
| QMSum | 14,258.40 | 13,052.0 | 30,354 | 0/40 |

실제 Input이 8192 Token을 넘는 요청은 172/200개, 86.0%다. 16384 Token을 넘는 요청은 95/200개, 47.5%다. 긴 Prefill 수요는 입력 분포에서 확인된다. 실제 P Compute 병목은 아직 이 데이터에서 측정하지 않았다.

## Context 처리

모델 Context Limit 후보는 40960 Token이다. 공통 Output Cap2048을 예약하여 Input Limit을 38912 Token으로 정했다. 기존 KV42GiB/Engine을 유지한다.

원본 200개 중 11개가 이 Input Limit과 Model Context Limit을 넘었다. 해당 11개는 앞 19456 Token과 뒤 19456 Token을 유지했다. 중간 구간을 제거했다. 원문 Context와 전체 Prompt는 별도로 보존했다.

이는 [공식 LongBench의 Head-tail 원칙](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/pred.py)을 따른다. 공식 코드는 Task Prompt를 먼저 자른 뒤 Chat Prompt를 만든다. 이번 구현은 정확한 최종 Token Budget을 보장하려고 Qwen Chat Template 뒤에 적용한다. 이 차이를 원본 정책과 완전히 동일하다고 표현하지 않는다.

전체 200개를 유지했다. Context 기준 Exclusion은 0/200개다. Truncation은 11/200개, 5.5%다. 반복 Padding이나 Filler는 없다.

GPU에서 최대 길이 요청의 Memory와 State Gate를 먼저 확인한다. 이 Gate가 실패하면 Main Sweep을 실행하지 않는다. Input Limit을 바꿔야 한다면 같은 Source ID를 유지하고 새 Manifest Version을 만든다. 기존 Manifest는 삭제하지 않는다.

## Natural Generation 계약

- Original BF16, Exact Causal Attention, Full36 Layers.
- Greedy. Thinking Disabled. EOS 허용.
- 공통 Max Output Cap: 2048 Token.
- Task별로 Output Cap을 바꾸지 않는다.
- 실제 Finish Reason과 Cap Hit를 저장한다.
- 실제 Output Length와 Input/Output Joint Distribution은 `NOT_GENERATED`다.

Natural Output 생성 뒤 Task별 Output Mean/p50/p90/p95/p99와 Cap Hit 비율을 추가한다. 이때 Source ID, 원문, Sampling, 순서를 바꾸지 않는다.

## Serving 계약

Main은 Agg DP2와 Native PD만 비교한다. Layer/Token Offloading과 Dynamic Scheduling은 실행하지 않는다.

모든 Engine에 B8192/S1024를 명시한다. Continuous Batching과 Chunked Prefill을 사용한다. 실제 Batch, Chunk와 KV Capacity는 별도로 기록한다.

Capacity Pilot 뒤 공통 QPS Grid를 정한다. 두 방법은 같은 Request Order, Poisson Arrival Seed와 Output Contract를 사용한다. Main 결과를 보기 전에 Grid와 중요 반복 조건을 동결한다.

제한적인 B4096/8192/16384 및 S256/1024 Sensitivity는 별도 표로 보고한다. 공통 설정 비교와 Best-tuned 비교를 혼합하지 않는다.

## Manifest와 SHA256

- [200개 Request Manifest](dense_ar_native_pd_real_workload_reproduction_20261119/data/longbench_v1_mixed200.json)
- [Sampling Freeze](dense_ar_native_pd_real_workload_reproduction_20261119/data/longbench200_sampling_frozen.json)
- [Truncation Manifest](dense_ar_native_pd_real_workload_reproduction_20261119/data/longbench200_truncation_manifest.json)
- [Task별 전체 통계](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/longbench200_pre_serving_contract.json)

Dataset SHA256: `1a9e6637bbd4f26e1338533d6735fa981c3ab258e0a59ac39c35b8b22d035553`.

Sampling Manifest SHA256: `68b45f70d182cc99985f800403a1a7fda626c4da7b717256eff225a791bd228a`.

Template Snapshot SHA256: `def665ed423ed549b4917991b7143b79e5c62e46c20538d3e2c0e3b4e52e6f11`.

이 Dataset은 다섯 Task의 동등 가중 Benchmark다. 실제 Production의 Request 분포를 대표한다고 주장하지 않는다. QA의 짧은 출력과 Summarization의 긴 출력은 현재 예상이며, 실제 Natural Generation으로 확인해야 한다.

## 2026-11-19 Natural Generation 추가 검증

이 절은 위 사전 계약을 변경하지 않는다. 같은 200개 Request의 실제 생성 결과를 추가한다.

상태: `INPUT_SELECTION_FROZEN / NATURAL_REFERENCE_COMPLETE / NATIVE_STATE_GATE_PENDING`.

Standard vLLM에서 Concurrency1로 Natural EOS를 측정했다. Original BF16과 동일 Model Revision을 사용했다. B8192/S1024, KV42GiB, FLASH_ATTN, HND를 명시했다. Batch-invariant 모드는 사용하지 않았다. 기본 Compile3과 `FULL_AND_PIECEWISE` CUDA Graph를 사용했다.

| 실제 Output 통계 | Token |
|---|---:|
| Mean | 128.675 |
| Min | 2 |
| p10 | 3 |
| p25 | 5 |
| p50 | 12 |
| p75 | 102.5 |
| p90 | 548.7 |
| p95 | 580.15 |
| p99 | 635.19 |
| Max | 657 |

| Task | Output Mean | Output p50 | Output p95 | Output Max |
|---|---:|---:|---:|---:|
| HotpotQA | 6.100 | 4.5 | 10.10 | 52 |
| MuSiQue | 8.700 | 5.0 | 32.05 | 38 |
| NarrativeQA | 8.175 | 8.0 | 15.05 | 16 |
| GovReport | 539.100 | 551.5 | 635.95 | 657 |
| QMSum | 81.300 | 76.5 | 121.00 | 152 |

Cap Hit는 0/200개다. 실제로 방출한 EOS Token을 Output Token 수에 포함했다. 이 결과는 독립 Server Start 1회의 출력 Reference다. Open-loop Serving Throughput 결과가 아니다.

최대 Input38912의 별도 Memory Control은 통과했다. 이것은 Native KV Handoff의 상태 정합성 통과를 뜻하지 않는다. Native 상태 검증은 별도로 수행한다.

- [실측 길이 통계](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/longbench200_natural_length_statistics.json)
- [실측 Output Manifest](dense_ar_native_pd_real_workload_reproduction_20261119/data/longbench200_measured_output_manifest.json)
- [Input/Output 분포](dense_ar_native_pd_real_workload_reproduction_20261119/figures/longbench200_length_distribution.png)
- [Input/Output Joint Scatter](dense_ar_native_pd_real_workload_reproduction_20261119/figures/longbench200_input_output_joint.png)

Natural Reference SHA256: `b7422c3a9a5ffafb6fa4cda9f340097faec428e92af7b39a3b3baf118cdaa51d`.
Input Dataset SHA256은 위 사전 계약과 같다. Sampling, Source ID와 Request Order는 변경하지 않았다.

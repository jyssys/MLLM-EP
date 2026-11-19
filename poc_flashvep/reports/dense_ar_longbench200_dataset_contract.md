# LongBench v1 Mixed200 Dataset 계약

상태: `MAIN_DATASET_PASS / MAIN200_OUTPUT_PASS`. Native Model 재현과 Concurrent Transport Tail의 판정은 별도 보고서에 기록한다.

200개 Source ID와 입력은 모델 Output을 보기 전에 동결했다. 최초 GPU 실행 전 계약은 [사전 Workload 계약](dense_ar_longbench_v1_200_workload_contract.md)에 보존했다. 이 문서는 실제 생성 결과를 추가한다.

## Source와 재현성

| 항목 | 값 |
|---|---|
| Dataset | [공식 zai-org/LongBench v1](https://huggingface.co/datasets/zai-org/LongBench) |
| Dataset Revision | `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` |
| Prompt Template Commit | `2e00731f8d0bff23dc4325161044d0ed8af94c1e` |
| Sampling Seed | `2026111999`에 Task Index0–4를 더함 |
| Mixed Order Seed | `2026112000` |
| Model | `Qwen/Qwen3-8B` |
| Model Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Precision / Attention | Original BF16 / Exact Causal Attention |
| Input Dataset SHA256 | `1a9e6637bbd4f26e1338533d6735fa981c3ab258e0a59ac39c35b8b22d035553` |
| Primary Natural Reference SHA256 | `838b8a88524921ae9165df95daa6299464e0e9c8eed0e6c2fa9c9653f7b9d019` |

각 Task의 원본200개에서40개를 균일하게 비복원 추출했다. 결과를 보고 요청을 교체하지 않았다. 서로 다른 Request ID는200개다. 서로 다른 Context Hash는163개다. 이전 Pool과 겹치는 Source ID18개를 보존했다. 전부 새로운 문서라고 주장하지 않는다.

## 실제 입력과 출력

실제 Token 수는 Qwen Chat Template 뒤에 측정했다. Thinking을 비활성화했다. 공식 LongBench의 단어 수 기반 평균 길이를 Token 수로 사용하지 않았다.

Primary 출력은 공식 vLLM의 Batch-invariant 모드에서 생성했다. Greedy와 Natural EOS를 사용했다. 공통 Cap은2048 Token이다. 모든200개가 EOS로 종료했다. Cap Hit는0/200이다. 방출한 EOS Token은 Output 수에 포함했다.

| Task | Requests | Input Mean | Input p50 | Input Max | Generated Output Mean | Output p50 | Output p95 | Output Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 40 | 13,366.975 | 14,689.0 | 17,305 | 6.100 | 4.5 | 10.10 | 52 |
| MuSiQue | 40 | 16,176.475 | 16,794.5 | 17,894 | 8.675 | 5.0 | 32.05 | 46 |
| NarrativeQA | 40 | 28,960.775 | 32,979.5 | 38,912 | 8.475 | 8.0 | 16.00 | 19 |
| GovReport | 40 | 10,607.675 | 8,838.0 | 38,912 | 541.525 | 528.0 | 651.90 | 717 |
| QMSum | 40 | 14,258.400 | 13,052.0 | 30,354 | 81.850 | 75.5 | 128.25 | 140 |
| Mixed200 | 200 | 16,674.060 | 16,021.0 | 38,912 | 129.325 | 12.0 | 605.10 | 717 |

| 통계 | Input Tokens | Generated Output Tokens |
|---|---:|---:|
| Min | 1,910 | 2 |
| p10 | 6,804.30 | 3 |
| p25 | 10,097.75 | 5 |
| p50 | 16,021.00 | 12 |
| p75 | 17,470.25 | 105 |
| p90 | 33,320.90 | 528 |
| p95 | 38,912 | 605.10 |
| p99 | 38,912 | 651.18 |
| Max | 38,912 | 717 |

총 Input은3,334,812 Token이다. 총 Output은25,865 Token이다. QA120개는 Output930 Token을 생성했다. Summarization80개는24,935 Token을 생성했다. 요청 비율과 Decode 작업량 비율은 다르다. 긴 입력만으로 D 여유를 확정하지 않는다.

BI0의 초기 Single-GPU Reference도 보존했다. 해당 Output Mean은128.675 Token, Max는657 Token이다. 두 수치 경로의 Output을 한 성능 비교에 섞지 않는다. Primary 비교는 같은 공식 Batch-invariant 경로와 같은 출력 Reference를 사용한다.

## Context와 Memory

Model Context Limit은40960 Token이다. 고정 Model Config의 `max_position_embeddings=40960`, `rope_scaling=None`을 확인했다. RoPE 확장 설정을 추가하지 않았다. Output Cap2048을 예약하여 Input Limit을38912 Token으로 정했다. GPU당 KV Budget은42GiB다.

200개 중11개를 Head19456+Tail19456으로 처리했다. NarrativeQA10개와 GovReport1개다. Exclusion은0개다. 원문과 전체 Chat Prompt, 제거 구간은 별도 Manifest에 보존했다. 반복 Filler는 사용하지 않았다.

공식 LongBench의 Head-tail 원칙을 참고했다. 공식 코드는 Chat Template 전에 자른다. 이번 구현은 최종 Token Budget을 맞추려고 Chat Template 뒤에 자른다. 이 차이를 기록했다. 원본 정책과 완전히 동일하다고 표현하지 않는다.

최대 Input38912에서 Output16을 실제로 생성한 Memory Control은 통과했다. 해당 GPU1의 Peak Allocated Memory는65,958,093,824 Byte였다. Native N8에서는 이 최대 길이의 Prompt KV와 Hidden 검증도 통과했다. 이는 모든200개 동시 실행의 Memory 통과를 뜻하지 않는다.

## Runtime과 GPU

- vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1.
- GPU0: `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`.
- GPU1: `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`.
- Primary B8192/S1024, Prefix Cache 비활성화, FLASH_ATTN, HND.
- 기본 Compile3과 `FULL_AND_PIECEWISE` CUDA Graph.
- Numerical 경로를 두 방법에 동일하게 적용한다. Model Weight나 Greedy Token을 덮어쓰지 않는다.

Source Hash와 실제 Worker Configuration은 각 Raw Job의 `source_hashes.json`, `configuration.json`, `actual_runtime_*.json`에 보존한다. Reference 생성은 Concurrency1이다. 이 출력 분포를 Online Throughput 또는 Sustainable Capacity라고 부르지 않는다.

## Manifest와 Figure

- [동결 Request Manifest](dense_ar_native_pd_real_workload_reproduction_20261119/data/longbench_v1_mixed200.json)
- [Sampling Manifest](dense_ar_native_pd_real_workload_reproduction_20261119/data/longbench200_sampling_frozen.json)
- [Truncation Manifest](dense_ar_native_pd_real_workload_reproduction_20261119/data/longbench200_truncation_manifest.json)
- [Primary Output Manifest](dense_ar_native_pd_real_workload_reproduction_20261119/data/longbench200_measured_output_manifest_BI1.json)
- [실제 길이 통계](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/longbench200_natural_length_statistics_BI1.json)
- [Input/Output Histogram](dense_ar_native_pd_real_workload_reproduction_20261119/figures/longbench200_length_distribution_BI1.png)
- [Joint Scatter](dense_ar_native_pd_real_workload_reproduction_20261119/figures/longbench200_input_output_joint_BI1.png)

이 Dataset은 다섯 Task의 동등 가중 Benchmark다. Production 대표 분포라고 주장하지 않는다. Prefill-heavy 여부는 같은 Online 실행의 P/D Queue와 GPU 계측으로 별도 판정한다.

## Input/Output 작업 구성의 추가 감사

다음 값은 실제 Token 수의 합이다. GPU FLOPs, 실행시간 또는 Headroom 측정이 아니다.

| 사전 정의 Stratum | 요청 | Input Tokens | Output Tokens | 전체 Input 비율 | 전체 Output 비율 |
|---|---:|---:|---:|---:|---:|
| QA | 120 | 2,340,169 | 930 | 70.17% | 3.60% |
| Summarization | 80 | 994,643 | 24,935 | 29.83% | 96.40% |
| Mixed | 200 | 3,334,812 | 25,865 | 100% | 100% |

긴 Input만으로 Mixed200 전체를 Prefill-heavy라고 판정하지 않는다. Summarization80개가 출력 Token의96.40%를 생성했다. 실제 P/D 병목은 같은 Online 실행의 Kernel Activity와 Queue로 확인해야 한다. [Token 구성 원시 계산](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/LongBench200_token_work_composition.json).

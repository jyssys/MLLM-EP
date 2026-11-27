# CogVLM2 two-H200 open-loop pilot

이 Pilot은 실제 원본 이미지를 처리하고 두 GPU에서 KV를 직접 이동하며, 요청별 Natural token 발생 시각을 기록한다. PyTorch Eager와 프로세스 내부 방출 경계의 Pilot이다. Stage의 KV와 방출 시점을 모델 상황에서 직접 통제할 수 있다.

## 동결한 비교

- GPU5/6 NV18, standalone-E가 아닌 E+P와 D. 모든 경우 원본 BF16 모델과 동일한 Vision attention 대응식, 같은 cohort와 Arrival Seed 2026112707.
- Co-located DP2는 GPU당 완전한 모델, 4 active slots, 최저 Outstanding Load를 사용한다. 각 시작에서 한 로컬 Prefill이 완료되면 곧바로 그 첫 Token을 냈다. 현재 FP forward는 작업을 중간에서 preempt할 수 없다.
- PD Resident는 original E/P와 fast Language-only D를 사용하고 visual bank를 D에 유지한다. PD Visual과 PD Reachability는 같은 실행에서 192개 D Weight를 제외한다. 두 후보의 Weight loader와 D 경로가 같은 GPU 작업을 수행한다.
- 모든 P+D에서 P가 계산한 첫 Token은 D에서 해당 KV가 준비되면 기록한다. KV copy는 확정된 모든 prompt layer를 실제 전송한다. D는 ragged batch의 original position 및 Mask를 유지한다.
- 기준선의 부하 Screening에서 Pilot QPS를 정했다. 8개 DP2의 0.5/1/2 offered req/s 부하에서 prespecified historical SLO 75% 및 finite-completion gate를 만족하는 가장 높은 수준을 선택한다. 조건을 만족하는 부하가 없을 때 0.5를 실행하는 절차를 미리 동결했다.
- historical joint SLO는 TTFT ≤5,000 ms, 모든 ITL ≤34 ms, TPOT ≤33 ms다. Natural EOS와 최대 2,048개의 cap을 사용했다. 이 Pilot의 요청 간 차이는 independent server start 종합을 대신할 수 없다. [동결한 기준](../mllm_architecture_first_disaggregation_discovery/results/online_pilot_contract.json)

첫 DP2 사전 테스트에서 여러 새 P 요청의 emission을 한 루프에 늦출 수 있는 경우를 코드 검토에서 확인했다. 로컬 Prefill마다 첫 Token을 바로 방출하는 Scheduler로 고치고 세 가지 부하를 다시 측정했다. 최종 비교는 수정한 경로를 사용한다. [실제 코드](../mllm_architecture_first_disaggregation_discovery/cog_online.py)

## 품질 해석

Natural ID 전체 일치와 공통 prefix 길이를 요청별로 측정한다. 시각 입력을 삭제하거나 파생 prompt로 대체한 요청은 없다. DocVQA의 local explicit-line ANLS, MMMU 및 기타 선택형의 명시 답안, decimal 및 일반 답안의 normalized exact를 로컬 보조 결과로 사용한다. Native는 실제 답변을 생성하며 sample selection에 정확도를 사용하지 않았다. 평가 Floor가 있을 수 있으므로 local metric과 정확한 token fidelity를 구분한다. Official scorer나 Judge가 필요한 품질의 비열등성은 별도 검증이다. [로컬 채점 계약](../mllm_architecture_first_disaggregation_discovery/analyze_cog_online.py)

## 실제 두 GPU 결과

수정한 8-request DP2 screening은 offered 0.5/1/2 req/s에서 각각 finite completion 0.400/0.590/0.583 req/s였다. 역사적 joint SLO를 지킨 요청은 각 1/8이었다. 동결한 fallback에 따라 0.5 기준으로 24개를 비교했다. 동일한 24개, offset, prompt hash, visual processor, cap과 GPU pair를 대조했다. 각 arm은 별도 신규 시작 1회이며 모든 요청이 EOS로 끝났다.

| 두 GPU Arm | TTFT p50/p95 ms | TPOT p95 ms | ITL p99 ms | E2E p95 s | Output tok/s | SLO req/s | SLO 요청 | Decode batch 평균/최대 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Co-located DP2 | 471.9 / 615.7 | 94.5 | 332.8 | 23.86 | 32.72 | 0.0366 | 2/24 | 1.285 / 3 |
| PD resident | 465.1 / 799.3 | 80.0 | 167.4 | 16.63 | 29.91 | 0.0373 | 2/24 | 1.606 / 4 |
| PD visual omission | 490.8 / 908.1 | 67.9 | 107.9 | 16.64 | 31.69 | 0.0386 | 2/24 | 1.423 / 3 |
| PD generic reachability | 482.1 / 1,016.2 | 65.8 | 182.3 | 16.57 | 30.72 | 0.0386 | 2/24 | 1.652 / 4 |

PD는 DP2의 긴 Decode gap을 줄였고 TTFT p95는 늘었다. Visual-pruned는 Resident PD보다 TTFT p95가 13.6% 길고, 두 prune 방식은 동일한 role/weight와 kernel을 사용했다. 각 arm에서 역사적 SLO에 맞춘 완료는 2건이라 Goodput의 작은 숫자 차이는 전체 완료 시간에도 좌우된다. 이 시작의 batch, kernel scheduling과 Natural length가 함께 달라져 modality-aware speedup을 분리할 수 없다. P/D 환산은 measured online이며 전 생산 workload의 지속 QPS로 확대하지 않는다. [페어 및 원시 시각](../mllm_architecture_first_disaggregation_discovery/results/cog_online/comparison.json)

D의 실제 Torch peak allocation은 Resident 32.025 GiB, Visual-pruned 18.259 GiB, Generic 18.934 GiB였다. P의 peak는 38.527–38.787 GiB였다. 세 P+D Arm의 전체 Prompt KV import 호스트 관측 중앙값은 2.20/2.11/2.21 ms였다. 최대를 포함한 분포와 실제 GPU 합계는 Raw에 있다. 이 Pilot의 active slot 상한은 GPU당 4다. 같은 배치 조건에서 저장 capacity를 높이는 실험은 별도 격리 측정이다. [실제 allocation과 per-request timing](../mllm_architecture_first_disaggregation_discovery/results/cog_online/visual_pruned.json)

Visual-pruned의 Natural output은 DP2, Resident PD, Generic PD와 각각 21/24에서 ID 전체가 일치했다. DocVQA 6/6과 MMMU 6/6이 각 Arm에서 일치했다. Local 점수는 4개 Arm에서 같고 MMMU/MMReason의 local 점수는 0이었다. 구조 차이를 분리한 deterministic teacher/per-layer 비교와 실사용 batch의 수치 변동을 함께 보존한다. 이 결과와 절대 SLO로 독립 N3, held-out 및 생산 runtime 후보 채택을 보류했다. [품질 및 수치 범위](mllm_architecture_disaggregation_correctness.md)

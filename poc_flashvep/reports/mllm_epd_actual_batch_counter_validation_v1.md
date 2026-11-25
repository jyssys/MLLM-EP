# 실제 Prepared Batch: Counter-only C3 검사

C46은 Agg DP3의 V2 Runner Counter API에서 실패했다. 실패 Raw와 종료 상태는 보존했다.
완료된 NativeStrong·NativeCommon·Static10·Idle10은 반복하지 않았다.
C46R1은 V2 Counter만 수정하고 Agg DP3·E+PD DP2를 새 초기화로 검사했다.
V2는 반환된 InputBatch와 중첩 EncoderRunner 출력을 읽었다. V1 Hook과 모델 실행은 유지했다.
각 Method는 같은 GPU4/5/6에서 새 초기화 1회로 실행했다. 독립 N3가 아니다.
동일 Main100·QPS2·요청별 Output 수를 사용했다. Natural EOS 결과와 합쳐 평균내지 않았다.
측정 순서는 Warm-up→Clean Before→Counter→Clean After다.
CUPTI·CUDA Event·추가 Tensor Copy·GPU 동기화는 사용하지 않았다.

## 계측 Gate

| Method | Observer≤5% | Complete geometry | Actual E outputs | Evidence |
|---|---|---|---|---|
| NativeStrong | False | True | True | DIAGNOSTIC |
| NativeCommon | False | True | True | DIAGNOSTIC |
| Static10 | False | True | True | DIAGNOSTIC |
| Idle10 | False | True | True | DIAGNOSTIC |
| AggDP3 | False | True | True | DIAGNOSTIC |
| E_PD_DP2 | False | True | True | DIAGNOSTIC |

전체 다섯 Metric의 Profile 차이와 Clean Spread가 각각 5% 이하여야 적격이다.
하나라도 실패하면 해당 Method의 계측은 DIAGNOSTIC이다. 유리한 Metric만 선택하지 않았다.
실제 Encoder 또는 Aggregated 역할의 이미지 수·Visual Row 합계도 원래 Main100과 일치해야 한다.
이 Gate를 통과해도 전체 Task Quality는 PENDING이다.

## Actual logical batch

| Method | GPU | Role | Prepared calls | Max requests | Max prompt requests | Max generation requests | Max query rows | Mean requests/call |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| NativeStrong | 4 | encoder | 0 | None | None | None | None | None |
| NativeStrong | 5 | prefill | 159 | 10 | 10 | 0 | 8192 | 1.5220125786163523 |
| NativeStrong | 6 | decode | 4241 | 4 | 1 | 4 | 4 | 1.242395661400613 |
| NativeCommon | 4 | encoder | 0 | None | None | None | None | None |
| NativeCommon | 5 | prefill | 161 | 5 | 5 | 0 | 8192 | 1.3850931677018634 |
| NativeCommon | 6 | decode | 4228 | 4 | 2 | 4 | 4 | 1.2462157048249765 |
| Static10 | 4 | encoder | 0 | None | None | None | None | None |
| Static10 | 5 | prefill | 152 | 8 | 8 | 0 | 8192 | 1.5263157894736843 |
| Static10 | 6 | decode | 5269 | 1 | 1 | 1 | 3054 | 1.0 |
| Idle10 | 4 | encoder | 0 | None | None | None | None | None |
| Idle10 | 5 | prefill | 157 | 8 | 8 | 0 | 8192 | 1.5477707006369428 |
| Idle10 | 6 | decode | 4539 | 4 | 2 | 4 | 2895 | 1.1608283762943379 |
| AggDP3 | 4 | agg | 1840 | 7 | 2 | 7 | 8192 | 1.2619565217391304 |
| AggDP3 | 5 | agg | 1076 | 6 | 2 | 6 | 8192 | 1.4349442379182156 |
| AggDP3 | 6 | agg | 1218 | 6 | 3 | 6 | 8192 | 1.2348111658456487 |
| E_PD_DP2 | 4 | encoder | 0 | None | None | None | None | None |
| E_PD_DP2 | 5 | pd | 1558 | 7 | 3 | 7 | 8192 | 1.7130937098844672 |
| E_PD_DP2 | 6 | pd | 1957 | 9 | 4 | 7 | 8192 | 1.3832396525293817 |

Prompt와 Generation Row는 실제 입력 준비 직전의 CPU Computed/Prompt 경계로 나눴다.
Scheduler가 먼저 Computed Count를 증가시켰다고 마지막 Prompt Chunk를 Decode로 바꾸지 않았다.
한 요청이 두 종류의 Query를 함께 포함하면 Mixed Request로 보존했다.
Native D의 첫 요청도 실제 준비한 Query 위치를 기준으로 분류했다. Stage 이름만으로 모든 Query를 Decode라 부르지 않았다.
E-only의 Scheduled Token 수를 LLM Forward 작업량에 더하지 않았다.
CUDA Graph의 Padding된 Kernel Shape와 Logical Query Geometry는 다르다.
빈 Worker Cleanup은 별도 Count다. GPU Prefill Service나 실제 요청 Batch1로 바꾸지 않았다.
평균은 Prepared Call 가중이다. 시간 가중 평균이나 요청 가중 평균이 아니다.
이 검사는 과거 Clean N3의 실제 Batch를 소급 측정하지 않는다.
모델·Compiler·Admission·Dataset·SLO는 원래 Family와 같다. Hook은 각 Counter 구간 후 원래 Method로 복원했다.
Exclusive Kernel Time·Queue Timeline·SM/HBM Counter·미래 Safe Window는 이 Counter의 측정 범위가 아니다.

## 재현 자료

Config는 configs/batch_COUNTER_C46.json이다. Worker별 원시 Histogram은 raw/batch_counter/batch_COUNTER_C46_B0_*/에 있다.
원시 Raw Metric 재계산·Histogram CSV·Lease SHA는 analysis/batch_COUNTER_C46_*_v0에 있다.
PNG/PDF와 대응 CSV는 figures/batch_COUNTER_C46_actual_batch_v0에 있다.
Original BF16 Qwen2.5-VL-7B Revision cc594898137f460bfe9f0759e9844b3ce807cfb5를 사용했다.
P/D·Replica B8192/S1024, KV42GiB, Block128, Prefix Cache Off와 원래 Vision·Processor를 유지했다.
이 결과는 새로운 Method의 성능 개선이나 독립 Serving GO가 아니다.

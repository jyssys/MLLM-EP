# MLLM EPD Encoder·Decoder Idle 추가 실행 계약

상태는 `PLAN_FROZEN`이다. 이 문서는 측정 결과가 아니다.
기존 MLLM EPD 캠페인을 유지한다. 추가 연구 예산은 최대 5시간이다.
전체 최대 예산은 시작 후 25시간이다. 마지막 2시간은 검증과 자원 복원에 우선 배정한다.

## 고정 조건

Qwen2.5-VL-7B-Instruct와 Original BF16을 사용한다.
Model·Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
이미지 전용 Main400을 변경하지 않는다. Video를 사용하지 않는다.
Main400 SHA256은 `7ac7d28ae1678499bee35ec20e317760555944cac7b4ca29eb46aeb3957b9592`다.
Calibration32·Discovery200·Heldout200의 이미지·문서 분리도 유지한다.

기본 구성은 같은 H100 3개의 E1P1D1이다.
P/D는 B8192/S1024를 명시한다. E는 공식 Encoder-only 경로의 B114688을 명시한다.
Actual Batch와 설정 상한을 구분한다.
Prefix Cache는 비활성화한다. Processor Cache 1 GiB와 요청별 고유 UUID를 모든 비교군에 적용한다.
이는 설치 Runtime의 엔진 간 Media ID 충돌을 방지하는 공통 정합성 수정이다.
고유 UUID 때문에 다른 요청의 Encoder 결과를 재사용하지 않는다.

## 실험 순서

1. 기존 Native EPD 정합성 Gate와 실제 Natural EOS 출력을 확인한다.
2. 같은 Trace에서 Low·Knee·High QPS를 선정한다.
3. 동일한 ROI에서 E/P/D Kernel·Copy·Queue를 계측한다.
4. Clock Alignment 오차와 Profiler 간섭을 검사한다.
5. E/D Idle Window와 동시 Idle을 분석한다.
6. E에 LLM Weight를 추가하는 메모리 적합성을 검사한다.
7. Gate를 통과한 경우에만 최소 Online Interference Probe를 수행한다.

기존 Trace가 충분하면 재사용한다. 새 QPS Sweep을 불필요하게 반복하지 않는다.
Main400이 원하는 P-heavy 상태를 만들지 못해도 요청을 교체하지 않는다.
정합성 실패 실행은 Primary 성능표에서 제외한다.

## 계측 계약

Kernel Busy는 GPU Kernel Interval의 합집합이다. Kernel Duration의 단순 합이 아니다.
Copy Activity는 별도 기록한다. Kernel이 없는 구간과 GPU 전체가 유휴인 구간을 구분한다.
NVML GPU Utilization을 SM Active·Tensor Core·HBM 측정으로 대체하지 않는다.
측정할 수 없는 Counter는 `NOT_MEASURED`로 남긴다.
Clean/Profile/Clean 간섭의 초기 상한은 주요 지표별 5%다.
실패한 계측은 `DIAGNOSTIC`으로 남긴다.

E/P/D는 같은 Host Monotonic Clock으로 정렬한다.
GPU Clock Mapping의 양 끝 Anchor와 오차를 보존한다.
짧은 Window의 길이가 정렬 오차와 비슷하면 실행 가능하다고 판정하지 않는다.
전체 Arrival→Drain과 Steady-state를 분리한다.
Steady-state의 주 구간은 두 번째부터 마지막 직전 예정 Arrival까지다.
요청 수가 적으면 해당 구간의 한계도 보고한다.

다음 상태를 별도 계산한다.

- P Active + E Idle
- P Active + D Idle
- P Active + E Idle + D Idle
- P Active + E Active + D Idle
- P Active + E Idle + D Active
- E/P/D All Active

분모는 전체 ROI와 P Busy Duration을 각각 사용한다.
P Queue가 존재한 시간과 P Kernel 실행 시간도 별도로 보존한다.
동시 Kernel Gap은 같은 요청의 3GPU 병렬 실행 가능성을 입증하지 않는다.

## Idle 유형과 Window

E의 Idle을 다음 범주로 분류한다.
근거가 부족하면 원인을 추정해서 채우지 않는다.

- `TRUE_ENCODE_IDLE`: Running·Pending Encode가 없음.
- `HOST_SCHEDULER_WAIT`: Encode 대기가 있지만 GPU 제출이 없음.
- `INPUT_OR_TRANSFER_WAIT`: Media 처리 또는 전달을 기다림.
- `INTRA_ENCODE_KERNEL_GAP`: Encode 실행 내부의 Kernel Gap.
- `DOWNSTREAM_BACKPRESSURE`: 명시한 Credit·P 의존성이 E를 막음.
- `UNRESOLVED`: 현재 Timestamp로 원인을 구분하지 못함.

범주별 시간·Count·평균·p50/p90/p95/p99·최대·시작 Pending 수·종료 원인을 보고한다.
Window Threshold는 0.1/1/5/10/20/50/100 ms로 동결한다.
Count 비율과 Idle Duration 가중 비율을 구분한다.

## Prefill Quantum과 안전성

Quantum은 128/256/512/1024/2048 Tokens를 우선한다.
같은 모델·Precision·Embedding·Prefix Length·Kernel Backend에서 E와 D를 비교한다.
Prefix Prefill과 기존 KV가 있는 Suffix Continuation을 구분한다.
Transfer·Admission·Launch·동기화 비용을 GPU Kernel 비용과 따로 보존한다.
원래 LLM Weight를 E에 추가한 뒤 Vision 용량·Workspace·Peak Activation·KV 여유를 측정한다.
독립 프로세스의 GPU Memory 합계만으로 공존 실행의 안전성을 판정하지 않는다.

Online Probe 전에 다음 Gate를 적용한다.

- Encoder Service p95 Ratio의 Paired 95% CI 상한 ≤1.05.
- Encoder Queue의 지속적 악화가 없음.
- Native-relative ITL p99·TPOT p95 CI 상한 ≤1.05.
- Absolute Decode Attainment ≥95%.
- 모델 정합성과 동일한 3GPU 예산을 유지함.

Vision Encoding을 우선한다. Stream Priority를 Kernel 선점 보장으로 취급하지 않는다.
추가 작업이 실제 요청 완료에 기여하지 않으면 `INTERFERENCE_PROBE`로 표시한다.
해당 Probe를 Serving Throughput 향상으로 보고하지 않는다.
미래 Idle 길이를 사용한 배치는 `OFFLINE_ORACLE`이다.
실제 추가 Prefill 없이 `ENCODE_SAFE_HEADROOM_CONFIRMED`를 부여하지 않는다.

## 범위와 후속 판단

Three-way Token Split·Context Parallelism·Layer Wavefront의 완전한 구현은 수행하지 않는다.
Request-level Offload·Inter-request Pipeline·Wavefront·Elastic Context Parallelism의 제약만 비교한다.
E Idle 활용 자체를 새 기여로 주장하지 않는다.
[EAServe](https://arxiv.org/abs/2609.31551)는 E와 Local Prefill의 자원 공유를 다룬다.
[HydraInfer](https://arxiv.org/abs/2505.12658)는 E/P/D 배치와 단계별 Batching을 다룬다.
[vLLM Context Parallel 문서](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/)는 Q/KV 분할과 통신을 설명한다.
이 자료들의 성능을 현재 Qwen2.5-VL C3 실측으로 대체하지 않는다.

최종 판정은 E Idle·E/D 동시 Idle·Encode-safe Headroom·후속 연구를 각각 구분한다.
실측·정합성·Oracle·미실행을 함께 보존한다.

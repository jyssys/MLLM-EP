# Generic Token Split — 동시 요청 Natural EOS 정합성

C33 N1과 N8의 Natural EOS 정합성 검사가 통과했다.
고유 요청은 8개다. N1 반복을 포함한 실행은 9회다.
모든 실행에서 전체 28개 Layer의 Prompt KV와 생성 Token이 비교 기준과 일치했다.
증거 수준은 `CORRECTNESS_ONLY`다. 이 Capture의 지연을 Online 성능으로 사용하지 않는다.

## 고정 계약

모델/Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
Qwen2.5-VL-7B, Original BF16, BI0, 원본 Vision Encoder와 Full28을 유지했다.
GPU4/5/6은 E/P/D다. P/D B8192/S1024, KV42GiB, Block128, Prefix Cache Off다.
공통 Compiled 설정은 기존 C30/C31 계약과 같다. Stock 기본 설정의 이전 실패는 보존했다.
원본 Calibration 이미지와 질문을 그대로 사용했다. Main100과 Main400은 변경하지 않았다.
마지막 20% Token을 D로 이전했다. Greedy, Natural EOS, Cap2048이다.
Teacher는 같은 Query Chunk 경계의 원본 E+P Full Prefill이다. C30의 원시 상태를 재사용했다.

## 동시 실행과 Admission

N8은 고정한 Poisson Arrival, QPS2.0으로 실행했다. E/P/D는 실제 요청과 실제 전송을 처리했다.
P의 실제 관측 Batch는 최대 4였다. D의 Prefill과 Decode Batch는 이 보수적 정책에서 1이었다.
추가 D Suffix는 이전 D HTTP Stream이 모두 끝난 뒤 시작했다.
HTTP Stream 수는 GPU Kernel Idle 또는 Decode-ready Queue의 직접 측정이 아니다.
첫 생성 Token에서 Prefill 보호 구간을 해제한다. Role-only SSE Header에서 해제하지 않는다.
이후 Native 요청은 기존 Decode와 함께 기본 Scheduler를 사용할 수 있다.
이 검사의 Static Token 요청은 다음 Suffix 전에 앞선 Decode 완료를 기다렸다.
보수적 Admission과 미래 Safe Window를 예측하는 Scheduler를 혼동하지 않는다.

## 상태와 출력 검사

실제 NIXL Prefix는 각 요청의 모든 28개 Layer에서 Byte 단위로 일치했다.
P의 0→k와 D의 k→L Query Coverage에는 누락과 중복이 없었다.
Prefill Query의 Position은 원본 MRoPE Slice와 일치했다.
생성 Token의 Position도 원본 Position Delta 공식을 사용해 매 Step 검사했다.
전체 Prompt KV 오차는 Prefix와 Suffix로 나눠 계산했다.
미리 고정한 허용값은 Relative L2≤0.02와 Cosine≥0.999였다.
실제 관측값은 두 영역 모두 0이었다. 즉 이번 실행은 허용값 이내 수준을 넘어 Byte 단위로 일치했다.

| Cohort | Calibration | Input | Natural Output | Teacher Output | Strict KV Layers | Strict Output |
|---|---:|---:|---:|---:|---:|---|
| N1 | 0 | 4831 | 16 | 16 | 28 | True |
| N8 | 12 | 5343 | 38 | 38 | 28 | True |
| N8 | 0 | 4831 | 16 | 16 | 28 | True |
| N8 | 31 | 27840 | 60 | 60 | 28 | True |
| N8 | 1 | 7823 | 20 | 20 | 28 | True |
| N8 | 20 | 17334 | 275 | 275 | 28 | True |
| N8 | 4 | 5032 | 74 | 74 | 28 | True |
| N8 | 6 | 2961 | 17 | 17 | 28 | True |
| N8 | 17 | 14431 | 9 | 9 | 28 | True |

45-Page 문서와 275-Token Natural Output을 포함했다. EOS를 조절하거나 출력을 덮어쓰지 않았다.
이 결과가 BI0의 모든 Batch Geometry에서 불변성을 보장하지는 않는다.
각 요청 완료 뒤 E Embedding ACK와 Lifetime을 검사했다. 전체 Ownership이 비어 있음을 확인했다.

## 남은 검증

CPU KV 복사, 저장과 Hash 계산이 실행에 개입했다. 이 실행의 TTFT/ITL/TPOT를 성능 표에 넣지 않는다.
Task별 점수는 보존했다. DocVQA Floor와 MMLongBench-Doc 답변 추출 미실행 때문에 전체 품질은 미확정이다.
Main100 비계측 N3, 강한 기준선 비교와 Decode 5% 비열등성은 별도 Gate다.
Prefill-heavy 조건, Encode-safe Headroom과 Active Decode 중 실행 가능한 Window도 이 결과가 입증하지 않는다.
대응 CSV는 `analysis/natural_token_C33_cases_v0.csv`와 `analysis/natural_token_C33_query_coverage_v0.csv`다.
전체 KV Tensor는 로컬 `raw/batch_prefix/`에 있다. Manifest와 SHA256으로 검증할 수 있다.

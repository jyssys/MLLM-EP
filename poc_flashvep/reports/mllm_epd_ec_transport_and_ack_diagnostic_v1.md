# MLLM Native EPD EC Transport and ACK Diagnostic

증거 수준은 DIAGNOSTIC_C3다. Generic Token Split이나 추가 Prefill 삽입은 실행하지 않았다.
GPU4/5/6에서 실제 E Encoding, 전체28-Layer Prefill, NIXL KV 전달과 Decode를 실행했다.
Model Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다. Original BF16과 BI0를 유지했다.
Main100, Processor, 입력, 이미지 순서, KV42GiB/Block128, P/D B8192/S1024를 유지했다.
E는 Cache0/Renderer2/B114688/S1024와 같은 원본 Vision Encoder를 사용했다.
Source Credit이나 인위적인 D Cap은 추가하지 않았다. 영상·Pruning·Layer/Hybrid는 사용하지 않았다.

## 실제 실행 상태

첫 CUDA IPC 후보 C7은 실행에 실패했다. 모델 출력 오류로 분류하지 않는다.
코덱이 메모리 Handle을 64바이트로 가정했다. 설치된 PyTorch가 반환한 Handle은 66바이트였다.
Opaque 바이트를 자르지 않고 보존하도록 수정했다. 범위·형상·BF16·Storage 경계 검사는 유지했다.
실패 Raw와 Source Archive는 보존했다. C7의 N8과 성능 측정은 NOT_RUN이다.
수정한 C8은 N1과 N8 Natural EOS를 기존 같은 Geometry 기준선과 일치시켰다.
실제 Mantis 요청1개의 MRoPE·입력 Embedding·Hidden과 전체28-Layer KV도 CPU 전달과 일치했다.
이는 전체 Cross-geometry 품질 Gate를 통과했다는 뜻이 아니다. 기존 품질 한계는 유지한다.
두 소비자와 비동기 Producer의 별도 전달8개 검사도 통과했다. 이는 Serving 속도 향상이 아니다.

## CPU 전달 대 On-demand GPU 전달

각 Arm은 독립 초기화1회다. 두 초기화를 독립 N3로 취급하지 않는다.
Replay는 매 Arm 5,269 Tokens다. Natural의 실제 출력량은 별도로 기록했다.
같은 도착·순서 Seed를 사용했다. 모든100개 요청을 Drain까지 측정했다.

| Method | Contract | Output Tokens | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | All-ITL PASS % | SLO output tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU staging | replay_q2 | 5269 | 7149.400 | 15.622 | 6.730 | 98.00 | 57.298 |
| CPU staging | replay_q3 | 5269 | 10071.165 | 17.516 | 6.931 | 100.00 | 70.072 |
| CPU staging | natural_q2 | 5284 | 7234.892 | 16.713 | 6.983 | 97.00 | 59.811 |
| On-demand CUDA IPC | replay_q2 | 5269 | 11544.585 | 14.384 | 6.800 | 99.00 | 39.038 |
| On-demand CUDA IPC | replay_q3 | 5269 | 18085.163 | 15.771 | 6.935 | 96.00 | 56.331 |
| On-demand CUDA IPC | natural_q2 | 5341 | 12489.678 | 14.689 | 6.825 | 98.00 | 46.932 |

GPU 전달은 값 정합성을 유지했지만 이 구현에서는 TTFT와 Goodput을 악화시켰다.
CPU 기준선을 약하게 바꾸거나 이 부정적 결과를 제거하지 않았다.
GPU 전달에는 E GPU Tensor 보유와 최대12GiB의 유효 Backing Storage 상한이 있다.
C8의 QPS3까지 관측한 Peak는 약4.024GB였다. 상한12GiB를 채워서 Admission이 막힌 결과는 아니다.
단위 GB와 GiB는 다르다. 전체 Driver Memory나 Activation Peak와 이 보유량도 다르다.

## ACK의 한 요소 인과 검사

복사 완료 뒤에도 P가 E의 ACK RPC 응답을 기다리는 경로를 확인했다.
File Arm은 GPU Clone 완료 동기화를 유지했다. 완료 Marker만 먼저 기록하고 P가 계산을 이어갔다.
E는 성공한 D 완료 시점에 Marker를 검사한 뒤 Tensor를 해제했다. 조기 Credit 반환은 하지 않았다.
동일 초기화에서 RPC→File→RPC를 실행했다. 이는 독립 Start3회가 아니다.

| Arm | TTFT p95 ms | Export RPC mean ms/load | Import/Clone/Sync mean ms/load | ACK mean ms/load | All-ITL PASS % |
|---|---:|---:|---:|---:|---:|
| A: rpc | 11813.866 | 48.041 | 155.432 | 48.064 | 100.00 |
| B: file | 9110.786 | 78.405 | 145.992 | 0.288 | 100.00 |
| C: rpc | 12490.034 | 48.561 | 157.951 | 49.321 | 99.00 |

File ACK에서 TTFT p95가 줄었다. 다시 RPC로 바꾸자 지연이 증가했다.
이 한 초기화는 동기 ACK 경로가 P Critical Path 지연에 기여함을 지지한다.
개별 요청이나 Load149개를 독립 Server Start로 취급하지 않는다. 인과 기여율의 일반적인 CI는 없다.
File ACK에도 CPU 기준선보다 높은 TTFT가 남았다. Export RPC와 Import·Clone·Sync 비용은 남아 있다.
Import·Clone·Sync Host Enclosure에는 필요한 E GPU 준비 대기가 들어갈 수 있다.
이를 순수 GPU 복사시간이나 제거 가능한 낭비 전부로 해석하지 않는다.

## Host 경계의 의미

600개 요청에서 서로 겹치지 않는6개 Host 구간의 합이 요청 TTFT와 일치했다.
이는 평균 TTFT의 가법 분해다. TTFT p95의 가법 분해나 요청별 GPU-exclusive 비용이 아니다.
E HTTP 완료는 두 Transport에서 같은 GPU-ready 경계가 아니다.
CPU 저장은 GPU 완료를 기다린다. GPU Handle 발행은 CUDA Event를 통해 실제 준비 완료를 기다릴 수 있다.
따라서 E HTTP 구간 감소를 Encoder GPU 계산 절감으로 해석하지 않는다.
P HTTP 구간에도 GPU·Scheduler·큐·Transfer·Control 비용이 함께 들어간다.
요청별6개 구간과 모든 실패 범주는 대응 CSV와 기존 Metrics JSON에 보존했다.

## 다음 Gate

Native full-P에서 Descriptor를 E 단계에 미리 발행하는 별도 한 요소 검사를 진행한다.
이 최적화는 Token-prefix에 그대로 적용할 수 없다. 각 이미지의 실제 Consumer Query Coverage 계획이 필요하다.
실제 Generic Token Split, Decode-safe 추가 작업, 독립 N3 및 전체 Task Quality는 아직 미완료다.
Prefill-heavy와 실제 E/D-safe Headroom도 이 검사만으로 확인하지 않는다.
CSV/PNG/PDF는 `mllm_epd_prefill_heavy_token_split_20261124/analysis/native_EC_*v1.*`와 대응 Figure에 있다.

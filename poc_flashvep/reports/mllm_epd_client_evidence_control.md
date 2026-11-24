# MLLM EPD Client Evidence Control

증거 수준은 DIAGNOSTIC_C3다. 같은 GPU4/5/6의 Native EPD를 사용했다.
Main100, QPS2, 도착 Seed2026112402, 순서 Seed2026112404를 유지했다.
요청별 동일 출력 길이를 재생했다. 각 Arm은 100개 요청과 5,269개 출력 Tokens를 완료했다.
이는 Natural EOS 결과가 아니다. 세 Arm은 독립 Start3회가 아니라 하나의 초기화를 공유했다.

## 원인 가설과 변경

완료한 요청의 큰 JSON 기록이 다른 요청의 토큰 수신을 막을 수 있다.
실제 Client 수신 함수의 최소 재현에서 80ms 기록 대기가 약81.6ms ITL을 만들었다.
이 값은 주입한 기록 비용의 CPU 재현이다. 실제 GPU 비용 측정값이 아니다.
Deferred Arm은 수신 중에 작은 완료 Journal을 기록했다. 전체 SSE 기록과 Prompt Token ID는 Drain 후에 기록했다.
전체 Raw Row 내용, 토큰 수신 시각, SLO 부등호는 바꾸지 않았다.
기본 Client 정책은 Immediate로 유지했다. 이후 CPU Renderer 비교에도 같은 Immediate 정책을 사용한다.
과거 Timestamp와 Raw Evidence는 수정하지 않았다.

| Arm | ITL p99 ms | Max ITL ms | All-ITL PASS % | TPOT p95 ms/token | TTFT p95 ms |
|---|---:|---:|---:|---:|---:|
| 0: immediate | 17.246 | 174.998 | 96.00 | 7.028 | 6597.964 |
| 1: deferred | 16.530 | 68.318 | 97.00 | 6.960 | 6965.793 |
| 2: immediate | 16.921 | 389.231 | 98.00 | 6.944 | 7025.275 |

Immediate의 All-ITL PASS는 96%와 98%였다. Deferred의 값은 그 사이인 97%였다.
따라서 이 실행은 All-ITL 보호의 독립적인 회복을 입증하지 못했다.
Deferred의 최대 ITL은 이 순서에서 더 작았다. 독립 반복 CI는 없으므로 일반적인 감소율을 확정하지 않는다.
N8 같은 Geometry에서 토큰과 Usage의 Strict Match를 통과했다. CPU 회귀 검사34개도 통과했다.
이 변경은 계측 경로의 기록 시점을 바꾼다. 모델 계산 절감이나 Generic Token Split의 Serving 이득이 아니다.
프로세스가 Drain 전에 강제 종료되면 두 큰 필드는 메모리에만 남을 수 있다. 완료 Journal의 지표는 따로 보존된다.
Primary 채택, Natural EOS 확인, 독립 Start 반복은 미완료다.
CSV: `mllm_epd_prefill_heavy_token_split_20261124/analysis/client_evidence_control_v0.csv`.
PNG/PDF: `mllm_epd_prefill_heavy_token_split_20261124/figures/client_evidence_control_v0.*`.

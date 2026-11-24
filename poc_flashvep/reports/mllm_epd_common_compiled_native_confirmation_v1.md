# Common Compiled Native EPD — 독립 N3 검증

공통 Compiled 설정의 Native EPD를 독립 서버 초기화 3회로 실행했다.
Main100 Natural EOS와 Equal-output Replay를 별도로 재계산했다.
증거 수준은 `MEASURED_ONLINE` Runtime이다. 전체 Task Quality는 `PENDING`이다.

## 계약과 재계산

Qwen2.5-VL-7B, Original BF16, BI0, 원본 Vision Encoder와 Processor를 유지했다.
Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다. GPU4/5/6은 E/P/D다.
B8192/S1024, KV42GiB, Block128, Prefix Cache Off, Natural Cap2048을 유지했다.
P/D 공통 Compiler 설정은 `mllm_epd_compiled_token_correctness_v1.md`에 있다.
Native에는 Token Adapter, 추가 Source Credit 또는 D Cap을 적용하지 않았다.
기존 Stock NativeStrong과 그 독립 N3 결과는 변경하지 않았다.
각 Start에서 N1 및 순차 N8 상태 검사를 다시 통과한 뒤 Timing을 시작했다.
Raw Token 수, Percentile과 Goodput 분자를 다시 계산했다. 저장된 Metric과 일치했다.
모든 요청 순서와 상대 Arrival은 같다. QPS JSON의 2와 2.0만 정규화해 Contract Hash를 비교했다.
원래 Arrival 파일은 수정하지 않았다. 모든 Start를 보존했다.

| Output | Start | Output Tokens | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Goodput tok/s | Joint % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Natural EOS | 1 | 5113 | 4493.311 | 16.097 | 6.926 | 89.245 | 97.000 |
| Equal-output replay | 1 | 5269 | 4660.348 | 16.261 | 6.909 | 93.066 | 95.000 |
| Natural EOS | 2 | 5396 | 5148.760 | 14.390 | 6.821 | 97.258 | 93.000 |
| Equal-output replay | 2 | 5269 | 4538.441 | 16.013 | 6.985 | 98.006 | 97.000 |
| Natural EOS | 3 | 5256 | 6266.307 | 15.424 | 6.896 | 66.875 | 77.000 |
| Equal-output replay | 3 | 5269 | 4512.823 | 15.960 | 7.004 | 91.753 | 96.000 |

## Start-mean과 변동성

| Output | TTFT p95 Start-mean ms | Start-mean 95% CI ms | ITL p99 Start-mean ms | Goodput Start-mean tok/s |
|---|---:|---|---:|---:|
| Natural EOS | 5302.793 | [4493.311, 6266.307] | 15.303 | 84.459 |
| Equal-output replay | 4570.537 | [4512.823, 4660.348] | 16.078 | 94.275 |

CI는 독립 Start 단위 Bootstrap이다. 300개 요청을 독립 서버 시작으로 취급하지 않았다.
N3의 CI 해상도는 제한적이다. Pooling한 p95와 Start별 p95 평균도 구분했다.
세 번째 Natural 실행의 TTFT p95는 6266.307ms였다. 첫 실행의 4493.311ms보다 높았다.
해당 실행을 제외하거나 유리한 결과가 나올 때까지 반복하지 않았다.
Natural Output은 Start별로 5113/5396/5256 Tokens였다. 실제 Batch와 BI0 Numerical Path에 따른 차이가 있다.
이 차이와 Queue 변동의 개별 인과적 기여율은 이 분석에서 확정하지 않았다.
Replay는 모든 Start에서 5269 Tokens다. Answer String의 동일성을 뜻하지 않는다.
Replay TTFT p95도 실제 관측 범위 그대로 보고했다. Natural과 합쳐 평균하지 않았다.

## SLO와 품질

TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms/token 계약을 유지했다.
Joint는 세 조건의 AND다. Goodput 분자는 Joint 통과 요청의 Output Tokens다.
분모는 첫 예정 Arrival부터 Drain까지다. Finite-cohort Throughput은 Sustainable Capacity가 아니다.
실패 8개 조합의 Request Count와 Output Token Weight를 각 Start/Output 계약별 CSV로 보존했다.
DocVQA Raw ANLS Floor와 MMLongBench-Doc 공식 답변 추출의 미실행은 여전히 남아 있다.
이 표로 전체 품질 PASS 또는 Token Serving 성공을 선언하지 않는다.
Raw/Derived 대응은 `analysis/common_native_C32_N3_summary_v0.json`에 있다.
CSV는 같은 Prefix의 `starts`, `failure_categories`, `request_lengths` 파일이다.
Figure는 `figures/common_native_C32_N3_start_variation_v0.{png,pdf,csv}`다.

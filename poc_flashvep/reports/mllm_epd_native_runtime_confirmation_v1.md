# Native EPD runtime confirmation — Main100, independent N3

실제 E/P/D Online 실행을 독립 서버 초기화 3회로 확인했다. 전체 Task Quality Gate는 미확정이다.
Token Split은 이 표에 없다. CPU 전송 경로 대비 독립 N3 개선율도 이 표로 판정하지 않는다.

GPU는 4/5/6이다. 모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
Original BF16, BI0, Full28, B8192/S1024, KV42GiB, Block128을 유지했다.
E는 원본 Vision Encoder를 실제 요청 도착 후 실행했다. E의 Token Budget은 114688이다.
Early P Descriptor와 완료 후 File ACK를 사용했다. Producer는 D 완료와 ACK 전까지 Tensor를 보존했다.

| Output contract | TTFT p95 ms | ITL p99 ms | TPOT p95 ms | Raw output tok/s | SLO-goodput tok/s | Joint % |
|---|---:|---:|---:|---:|---:|---:|
| Natural EOS | 4880.240 | 16.267 | 6.827 | 98.239 | 82.523 | 91.667 |
| Equal-output replay | 5174.929 | 15.918 | 7.030 | 98.990 | 82.062 | 86.333 |

표의 값은 Start별 Metric의 평균이다. 300개 요청을 Pooling한 Percentile이 아니다.
각 Start의 수치와 Start-mean Bootstrap 95% CI는 대응 JSON에 보존했다.
Natural EOS와 Replay의 평균은 서로 합치지 않았다. Replay는 각 Start에서 5269 Output Tokens다.
모든 Start의 상대 Arrival Trace와 요청 순서는 같았다. 절대 실행 시각은 다르다.

TTFT SLO는 5000ms다. 모든 Client ITL은 34ms 이하여야 한다. Request TPOT는 33ms/token 이하여야 한다.
Joint SLO는 세 조건의 AND다. Goodput은 Joint 통과 요청의 Output Tokens를 전체 측정 기간으로 나눈 값이다.
측정 기간은 첫 예정 Arrival부터 마지막 Drain까지다. Finite-cohort Throughput은 Sustainable Capacity가 아니다.

CUPTI Profile은 독립 성능 측정과 분리했다. C11 S2 Profile은 5% 계측 기준을 통과하지 못했다.
따라서 해당 Kernel Busy와 Idle 수치는 진단용이다. Decode-safe Headroom은 아직 미검증이다.
MMLongBench-Doc의 공식 GPT 답변 추출은 미실행이다. Raw Answer 점수를 완전한 공식 품질 평가로 사용하지 않는다.

대응 CSV: `analysis/native_inline_independent_N3_v0.csv`.
대응 JSON: `analysis/native_inline_independent_N3_v0.json`.

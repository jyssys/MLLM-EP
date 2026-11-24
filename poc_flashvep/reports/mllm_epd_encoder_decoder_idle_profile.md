# E/P/D Idle Profile: MLLM100 Diagnostic

이 결과는 `DIAGNOSTIC` Online 실행의 `TRACE_ANALYSIS`다.
Model Quality와 Observer Gate가 아직 통과하지 않았다. Primary GPU 시간으로 사용하지 않는다.
Native EPD는 Original BF16 Qwen2.5-VL-7B와 실제 Online Encode를 사용했다.
GPU0/1/2는 각각 E/P/D다. GPU UUID와 Source Hash는 대응 Server Config에 있다.
P/D B8192·S1024, KV42GiB, Block128을 사용했다. E의 B는 114688이다.
현재 Main100 SHA256은 `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1`이다.
QPS는 1.0이다. Arrival Seed는 2026112402, Order Seed는 2026112404다.
요청별 기존 Natural 출력 길이를 고정해서 재생했다. 이는 Natural EOS 성능 결과가 아니다.

## 중앙 Arrival 구간

분모는 Arrival p10부터 p90까지 78.586494초다.
이 구간을 Sustainable Steady State로 확정하지 않는다. 전체 Cohort 결과도 JSON에 따로 있다.

| Stage | Kernel Busy | Kernel Inactivity | No Pending HTTP + Kernel Idle | Pending HTTP + Kernel Idle |
|---|---:|---:|---:|---:|
| encoder | 25.303% | 74.697% | 46.457% | 28.240% |
| prefill | 29.343% | 70.657% | 46.371% | 24.286% |
| decode | 25.059% | 74.941% | 50.829% | 24.111% |

Kernel Busy는 Kernel 구간의 Union이다. SM Active·Tensor Core Activity·HBM 포화율이 아니다.
GPU Copy·Memset 구간은 별도로 보존했다. 이를 Prefill 실행 가능 시간에 그대로 더하지 않는다.
Pending HTTP는 전처리·Transfer·Scheduler Queue·GPU Service를 모두 포함한다.
따라서 Pending Idle을 Host Wait나 Transfer Wait 중 하나로 단정하지 않는다.
현재 중앙 구간의 P Busy는 80% 기준에 못 미친다. 이 QPS에서 Prefill-heavy는 미입증이다.

## Observer 확인

| Metric | Profile / Clean 평균 차이 | 두 Clean 사이 상대 차이 |
|---|---:|---:|
| duration_seconds | 0.171% | 0.662% |
| TTFT_p50_ms | -2.870% | 4.884% |
| TTFT_p95_ms | 1.056% | 12.786% |
| ITL_p99_ms | 0.390% | 22.627% |
| TPOT_p95_ms | 2.879% | 10.238% |

Profile과 두 Clean 평균의 차이는 각 Metric에서 5% 이내였다.
그러나 Clean 사이의 TTFT p95·ITL p99·TPOT p95 차이가 5%를 넘었다.
따라서 안정적인 Baseline을 확인하지 못했다. Observer Gate는 통과하지 않았다.
이를 Profiler의 인과적 간섭이 5%를 넘었다는 확정으로 바꾸지 않는다.
CUPTI Dropped·Invalid Record는 세 Stage 모두 0이다.
SM Active·Tensor Core Activity·HBM Bandwidth는 `NOT_MEASURED`다.
Encode-safe·Decode-safe 추가 Prefill 실행도 `NOT_RUN`이다.

Raw Profile: `raw/profiles/native_main100_CPC_q1_v0/`
집계: `analysis/native_main100_CPC_q1_v0_idle_analysis.json`
Window CSV: `analysis/native_main100_CPC_q1_v0_idle_windows.csv`

# Native EPD의 Encoder 자원과 Decode Attention

## 측정 계약

기존 NativeStrong Free 1E/1P/1D를 그대로 시작했다. Qwen2.5-VL-7B-Instruct BF16 revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, vLLM `0.26.0+cu129`, H200 GPU4/5/6와 NV18 연결을 사용했다. Vision embedding은 CUDA IPC, Prompt KV는 NIXL, 클라이언트는 HTTP SSE 토큰 ID를 사용한다. Decode admission은 Free, stage는 기존 compiled / graph 설정이다. 동일 Main100에서 정해진 순서와 Poisson seed를 재사용했고 다른 다섯 GPU의 train_lm.py는 유지했다.

| Offered req/s | 완료 | 유한 cohort req/s | TTFT p95 | TPOT p95 | ITL p99 | Joint SLO |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 100/100 | 1.025 | 2.784 s | 5.065 ms | 9.003 ms | 99% |
| 2.5 | 100/100 | 1.881 | 8.112 s | 5.095 ms | 9.753 ms | 68% |
| 5.0 | 100/100 | 2.932 | 12.784 s | 5.522 ms | 12.703 ms | 54% |

역사적 SLO는 각 요청의 TTFT ≤5,000 ms, 모든 ITL ≤34 ms, TPOT ≤33 ms다. 세 부하 중 QPS1만 95% JOINT 통과했고 QPS2.5/5는 큐가 형성됐다. 처리량은 정해진 첫 도착에서 전체 완료까지의 값이다. [Native scan JSON](../encoder_assisted_decode_attention/results/native_scan.json) · [부하별 latency 및 goodput](../encoder_assisted_decode_attention/figures/native_qps_screen.png)

## 시간적으로 일치하는 기회

Native scan이 직접 기록한 중앙 20–80% 구간에서, E HTTP 요청이 하나도 없었던 시각은 D HTTP 활성 구간의 QPS1 **63.85%**, QPS2.5 **24.97%**, QPS5 **3.51%**와 겹쳤다. QPS1의 E HTTP p95는 **1.195 s**, QPS2.5에서는 **5.847 s**였다. 이 envelope에는 host dispatch, 대기, device 실행 및 응답이 모두 포함돼 있다. QPS1의 이 수치는 candidate의 사전 부하 선택과 host queue diagnosis에만 사용한다. GPU의 실제 동시 kernel과 미래 slack은 아래 별도 관측의 범위에 속한다. [Host timeline 지표](../encoder_assisted_decode_attention/results/native_scan.json)

CUPTI에서는 E/P/D의 clock anchor가 같은 host monotonic 시계와 중앙 도착 구간을 덮는지 검증한다. D에서 실제 FA3 CUDA kernel명에 해당하는 forward와 split combine을 찾아 idle E 및 E HTTP pending이 없는 실제 구간과 교차한다. 집계하는 시간 구간은 kernel wall-time union이다. QKV, O projection, FFN의 일반 GEMM 이름이 같아 해당 경계를 CUPTI kernel명만으로 분리할 수 없다. HBM은 별도 NVML GPM의 hardware DRAM bandwidth utilization, SM, Tensor, PCIe 및 NVLink metric으로 측정한다. GPM interval은 microsecond kernel의 HBM bytes를 직접 보여 주지 않는다.

같은 200-arrival trace의 clean/profile/clean에서 throughput duration, TTFT p50/p95, ITL p99 및 TPOT p95를 검증하고 모든 편차와 clean spread에 5% 제한을 적용한다. QPS2.5는 QPS5 대비 더 적게 포화되면서 P/E의 위상 경합이 나타난 stress point다. GPM은 profiler arm에서만 활성화해 두 관측의 합친 영향을 비교한다. 각 attention launch의 보조 실행에 필요한 미래 gap은 정해진 guard의 사후 관측값으로 그린다. 실시간 scheduler의 deadline-safe 승인률은 candidate가 실제 동작하는 온라인의 범위다.

## 메모리 범위

Engine 기동 후 Torch live allocation은 E 2.04 GiB, P/D 각각 57.90 GiB였다. 파라미터 payload는 E 1.26 GiB, P/D 각각 15.45 GiB였고 고유 live KV는 E 0, P/D 각각 42.00 GiB였다. E의 CUDA device free 137.00 GiB는 worker 기동 뒤 일시 계측이다. Safe KV 평가는 같은 E가 요청을 처리하는 동안 실제 cuda arena를 예약하고 페이지를 초기화하는 방법으로 별도 검증한다.

P/D 레이어의 28-layer BF16 KV는 4 KV heads, head dimension 128, K/V 두 개로 row당 57,344 bytes다. 따라서 아래에서 확인한 64 GiB는 분석상 1,198,372 KV rows에 해당한다. 실제 admission에는 block padding, Local/Remote 분할비, runtime workspace와 assistance overhead가 함께 작용한다.

## 재현

Github checkout `h200-eada`에서 기존 저장소와 hash-locked bootstrap receipt를 참조한다. 첫 Native scan 시 startup sparse-checkout의 이전 공용 모듈이 누락됐고 실패 receipt를 보존했다. 같은 branch의 sparse scope에 기존 의존 경로를 추가한 후 독립 run으로 전 표를 완료했다.

```bash
export PYTHONDONTWRITEBYTECODE=1
epd_python=../mllm_epd_runtime_env/bin/python
asset_repo=../MLLM-EP-mllm-epd
asset_receipt=../mllm_epd_autosetup/download_ready_1795600931728841878.json
"$epd_python" -m poc_flashvep.mllm_epd_h200_wide_decode_batch_token_split.online \
  --asset-repo "$asset_repo" --receipt "$asset_receipt" --run-dir ../h200_eada_runs/native_scan_retry \
  --mode native-strong --decode-cap free --qps 1 2.5 5 --defer-client-raw --wall-seconds 1800
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.native_profile \
  --asset-repo "$asset_repo" --receipt "$asset_receipt" --run-dir ../h200_eada_runs/native_profile_q2p5 \
  --mode native-strong --decode-cap free --qps 2.5 --profile --short-roi --profile-state-ms 200 \
  --profile-decode-buffer-gib 2 --main100-cycles 2 --gpm-period .4 --reserve-encoder-gib 96 \
  --defer-client-raw --wall-seconds 1500
```

GPU lease는 기존 소유권 확인을 사용하며 실행 경로가 다른 프로세스를 종료하지 않는다. 측정 후 기존 GPU burner를 복원한다.

[Native run provenance ledger](../encoder_assisted_decode_attention/results/native_resource_ledger.json)에 GPU UUID와 NUMA 배치, runtime과 runner fingerprint, cohort별 입력 및 원본 결과 SHA-256, 실패 receipt를 저장했다. 경로는 원본 `h200_eada_runs`의 상대 경로다. 이 ledger는 원본에서 다음 명령으로 재생성할 수 있다.

```bash
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.native_resource_ledger \
  --raw-root ../h200_eada_runs --out /tmp/eada-native-resource-ledger-repro.json
```

## QPS2.5의 실제 kernel 및 HBM 관측

동일 200-arrival의 Native clean → CUPTI/GPM → clean 세 번이 각각 200/200 완료됐다. CUPTI의 E/P/D clock, 중앙 구간 coverage, 누락 검사는 통과했다. 아래 표처럼 instrumented arm의 세 지표가 역사적 **5%** 범위를 벗어나 전체 profiling gate는 **FAIL**이다. 이 절의 수치는 직접 계산한 **DIAGNOSTIC**이며 Gate 0 또는 scheduler 승인률에 대한 근거는 보류한다.

| 측정 | Clean before | CUPTI/GPM | Clean after | Profiler–clean 중앙 상대 편차 |
|---|---:|---:|---:|---:|
| TTFT p50 | 3.468 s | 3.262 s | 3.325 s | 3.96% |
| TTFT p95 | 8.193 s | 7.482 s | 8.096 s | 8.14% |
| ITL p99 | 12.295 ms | 13.352 ms | 12.247 ms | 8.81% |
| TPOT p95 | 5.243 ms | 5.980 ms | 5.361 ms | 12.79% |
| 첫 arrival부터 drain | 82.952 s | 84.085 s | 82.600 s | 1.58% |

직접 clock-align한 중앙 도착 **42.624 s**에서 E/P/D kernel-busy는 **70.10/83.38/50.58%**였다. D의 FA3 forward/merge는 **D GPU-busy 구간의 11.32%**, D HTTP 요청들의 union의 **7.14%**였다. 조건부 worker sample 95개의 준비 Decode schedule은 batch p50/p95/max **1/3/6**, 합친 Prompt 길이 p50/p95 **11,353/29,587 rows**였다. 전 kernel에 20%의 attention latency 감소가 적용돼도 전체 D GPU busy 개선의 낙관적 한계는 **2.26%**다. 이 분모는 실제 GPU 전체 활동이며 지속적인 TPOT 대용 값은 별개다.

같은 95개 sample의 합친 Prompt 길이 최대는 **45,199 rows**였다. 별도 물리 attention에서 시험한 B32×8K, B16×16K, B8×32K는 약 262,144 prompt rows와 신규 row로 준비했으며 20% 이상 단축됐다. 실제 ragged B4의 60,949 KV rows에서는 가장 빠른 Generic 분할도 9.31% 느렸다. 이 관측은 병렬화 crossover와 해당 Native 부하에서 보이는 대다수 준비 geometry 사이의 간격을 보여 준다. 표본 밖의 모든 step에 대한 임계값으로 일반화하지 않는다. [물리 FA3 crossover](mllm_encoder_assisted_decode_attention_microbenchmark.md)

| 실제 D attention kernel 시간의 교집합 | 비율 |
|---|---:|
| E 물리적 kernel idle | 30.29% |
| E 물리적 idle, E HTTP 미처리 없음 | 11.63% |
| P 물리적 idle, P HTTP 미처리 없음 | 5.24% |

보수적으로 E HTTP까지 비어 있었던 실제 시각에서 향후 **500 µs**가 남은 D attention launch는 **13.70%**였다. 이 수치는 보조 실행이 발생하지 않은 원본 trace의 사후 fit이다. 실제 EADA가 arrival 시점의 미래를 알 수 없다. D-time의 위 교집합은 시각에 정렬한 구간에서 계산했다. E idle과 D busy의 개별 비율에는 곱셈을 적용하지 않았다. [Native 동시성](../encoder_assisted_decode_attention/figures/native_encoder_opportunity.png) · [E/P/D timeline](../encoder_assisted_decode_attention/figures/native_epd_timeline.png)

NVML GPM Encoder DRAM BW utilization은 중앙 구간 시간가중 평균 **21.49%**, p95 **32.64%**였다. D attention과 겹쳤던 0.4초 수준의 counter window에서 E DRAM utilization의 교집합 시간가중 평균은 **21.93%**였다. 같은 중앙 구간 E의 SM active 평균은 **68.56%**, Tensor active는 **23.78%**다. E는 활용 가능한 대역폭을 시사하지만 SM과 텐서 파이프라인에 원격 작업을 배치할 때의 지연과 간섭은 Online 검증에 남는다. P의 동시 window 평균 DRAM utilization은 **16.75%**이면서 idle과 HTTP 미처리 없음이 동시에 성립한 비율은 더 낮았다. 장치 수준 window들의 percentile은 개별 커널 HBM byte counter가 아니다.

Warm 이후 E Torch live **2.925 GiB**, reserved **37.389 GiB**, 계획 내 allocated high watermark **18.638 GiB**, NVML peak device used **38.748 GiB**, CUDA device free **101.653 GiB**였다. 따라서 첫 원본 E worker의 free 137 GiB를 지속 KV headroom으로 해석하면 실제 warm allocator footprint를 놓친다. 동일 snapshot의 P/D reserved는 약 **61.105/61.104 GiB**, CUDA free는 **77.642 GiB**였다. Probe launcher의 첫 호출이 Python stdlib `operator` shadowing으로 CUDA 초기화 전에 실패했다. 실패 raw를 유지하고 패키지 `python -m` child 진입을 검증했다. 실패 회차에는 96 GiB를 실제 할당하지 않았다.

200개 NIXL decode connector 전송의 payload는 총 **127.217 GB**였다. Transfer 시간 합계 **3.506 s** 또는 전송당 mean **17.531 ms**, post 시간 전송당 mean **10.440 ms**였다. 요청 사이에서 중첩된 전송이 있을 수 있으므로 전송 시간 합계는 Host timeline의 경과 시간이 아니다. 계측되는 payload 및 시간은 Native P→D connector 범위다. [원시 지표와 하드웨어 Counter](../encoder_assisted_decode_attention/results/native_profile.json) · [GPU 메모리 분해](../encoder_assisted_decode_attention/figures/native_epd_memory.png)

```bash
study=poc_flashvep/encoder_assisted_decode_attention
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.motivation_analysis \
  --run-dir ../h200_eada_runs/native_scan_retry --native-scan --out "$study/results/native_scan.json"
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.motivation_analysis \
  --run-dir ../h200_eada_runs/native_profile_q2p5 --tag native_strong_free_2p5_sustained_r2_profile \
  --out "$study/results/native_profile.json" --timeline-csv "$study/results/native_profile_timeline.csv"
matplotlib_python=/home1/irteam/ddn/envs/kvp_tf453/bin/python
"$matplotlib_python" -m poc_flashvep.encoder_assisted_decode_attention.plot_motivation \
  --results "$study/results" --out-dir "$study/figures"
```

분석의 결과 파일은 새 경로를 요구한다. 실행 결과 전체는 외부 `h200_eada_runs`에 보존했고 커밋에는 시각화와 재현에 필요한 경량 통계를 사용했다.

## 동일 Native EPD 부하에서 물리적으로 확인한 64 GiB

별도 NativeStrong 시작에서 QPS2.5, 동일 100개 Main100과 seed의 open-loop 실행을 두 번 완료했다. 첫 회차 종료 시 E/P의 실제 CUDA free는 **97.177/77.653 GiB**, Torch reserved는 **41.879/61.105 GiB**였다. 실험에서 항상 남기기로 한 최소 4 GiB free를 만족하지 못해 **96 GiB는 할당 전에 거절**했다. 해당 실행의 E는 관측 전의 E와 같은 BF16 모델 및 GPU였지만 뒤의 warm allocator 상태가 달랐다.

동일 시작에서 E GPU에 **64 GiB**를 물리 할당하고 초기화 커널로 전체를 썼다. Arena를 별도 프로세스에 유지하면서 원래 Encoder와 P/D를 함께 실행했다. 64 GiB 보유 동안 모든 100건이 완료됐고 실패 또는 OOM은 0건이었다. E의 NVML peak device memory는 **43.225→107.828 GiB**, resident arena를 가진 상태의 worker CUDA free는 **32.574 GiB**였다. Native Encoder allocator의 high-water allocated는 18.701 GiB로 같았고 reserved는 41.879→41.881 GiB였다. 이 결과는 해당 입력 및 프로세스의 실제 co-residency lower bound다. Arena는 모델 KV와 같은 byte capacity만 보유했으며 실제 KV admission이나 추가 E compute의 수용 능력은 여기서 측정하지 않았다.

| 동일 도착 100건 | Native EPD | E에 물리 64 GiB 추가 보유 |
|---|---:|---:|
| 완료 | 100/100 | 100/100 |
| TTFT p95 | 7.618 s | 7.321 s |
| TPOT p95 | 5.054 ms | 5.139 ms |
| ITL p99 | 10.922 ms | 11.557 ms |
| Encoder HTTP p95 | 5.268 s | 4.923 s |
| Joint SLO | 72/100 | 71/100 |
| SLO goodput | 77.03 tok/s | 76.63 tok/s |

양쪽에 공통으로 같은 E GPM sampling을 적용했다. ITL p99가 한 시작 안의 두 실행에서 약 5.81% 변했다. 물리 64 GiB의 존재를 latency neutrality나 최대 admission 이득으로 해석할 실험은 아직 부족하다. 같은 시작의 P에서 실제 보인 여유는 E보다 19.52 GiB 낮았다. 이 비교는 고정 42 GiB native P KV 예약을 포함하고 P에 별도 arena를 쓰지는 않았다. 실제 일반 LLM의 P-assisted와의 비교는 별도 실험이다. [물리 할당 및 matched Trace 통계](../encoder_assisted_decode_attention/results/native_capacity_retry.json) · [메모리 동시 상주 Figure](../encoder_assisted_decode_attention/figures/native_encoder_capacity.png)

```bash
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.native_profile \
  --asset-repo "$asset_repo" --receipt "$asset_receipt" --run-dir ../h200_eada_runs/native_capacity_retry \
  --mode native-strong --decode-cap free --qps 2.5 --capacity-only --main100-cycles 1 \
  --reserve-encoder-gib 96 --fallback-encoder-gib 64 --defer-client-raw --wall-seconds 1050
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.motivation_analysis \
  --run-dir ../h200_eada_runs/native_capacity_retry --capacity-study --out "$study/results/native_capacity_retry.json"
```

## Encoder 80 GiB 및 Prefill 64 GiB 교차 확인

각각 독립적인 NativeStrong 시작에서 동일한 100개 요청의 QPS2.5 도착 trace를 실행했다. BF16, GPU4/5/6, P/D의 42 GiB KV 설정과 최소 4 GiB 물리 여유 기준을 유지했다. E의 물리 80 GiB와 P의 물리 64 GiB를 각각 초기화해서 두 번째 회차 내내 유지했고 양쪽에서 100/100을 완료했다. P의 80 GiB는 stock warm 이후 E/P의 물리적 차이를 직접 확인하도록 같은 기준으로 신청했으며, 84 GiB의 요구 여유를 충족하지 못해 사전에 거절됐다.

| 실제 장치 | Warm 이후 CUDA free | Arena held | Peak 장치 사용량 기본 → arena | Arena 유지 시 종료 뒤 CUDA free | 결과 |
|---|---:|---:|---:|---:|---|
| E (GPU4) | 104.724 GiB | 80 GiB | 35.678 → 116.281 GiB | 24.121 GiB | 100/100 |
| P (GPU5) | 77.653 GiB | 64 GiB | 62.748 → 127.349 GiB | 13.052 GiB | 100/100 |

측정한 추가 arena 용량의 하한에서 E에 16 GiB 차이가 있다. P가 기존 prompt KV를 직접 유지하면 별도 구조로 같은 P 메모리를 사용할 수 있다. 이번 물리 비교는 두 역할에서 당시 Native allocator와 함께 보유 가능한 추가 공간을 확인한다. Assisted KV 운영, 추가 최대 동시 요청 수 및 일반 LLM의 최적 2P/1D 대비 이득은 별도의 실험이다. Arena가 새 CUDA 커널을 내지 않는 상태에서도 E80 회차의 ITL p99는 **9.044→10.906 ms**, P64 회차는 **10.470→9.988 ms**로 달랐다. 독립 N1에서 관측한 성능 변동이다. [E80 원본 집계](../encoder_assisted_decode_attention/results/native_encoder_capacity80.json) · [P80 거절 및 P64 원본 집계](../encoder_assisted_decode_attention/results/native_prefill_capacity.json) · [역할별 물리 reserve](../encoder_assisted_decode_attention/figures/native_encoder_prefill_capacity.png)

```bash
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.native_profile \
  --asset-repo "$asset_repo" --receipt "$asset_receipt" --run-dir ../h200_eada_runs/native_prefill_capacity \
  --mode native-strong --decode-cap free --qps 2.5 --capacity-only --capacity-role prefill \
  --main100-cycles 1 --reserve-arena-gib 80 --fallback-arena-gib 64 --defer-client-raw --wall-seconds 1050
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.motivation_analysis \
  --run-dir ../h200_eada_runs/native_prefill_capacity --capacity-study --out /tmp/eada-native-prefill-capacity.json
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.native_profile \
  --asset-repo "$asset_repo" --receipt "$asset_receipt" --run-dir ../h200_eada_runs/native_encoder_capacity80 \
  --mode native-strong --decode-cap free --qps 2.5 --capacity-only --capacity-role encoder \
  --main100-cycles 1 --reserve-arena-gib 80 --defer-client-raw --wall-seconds 1050
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.motivation_analysis \
  --run-dir ../h200_eada_runs/native_encoder_capacity80 --capacity-study --out /tmp/eada-native-encoder-capacity80.json
```

## 관측 빈도를 낮춘 동일 5% qualification

동일 QPS2.5 200-arrival에서 sampler의 GPM 주기를 0.4초에서 **1.2초**, worker 상태 수집 간격을 200ms에서 **1,000ms**로 늘렸다. E/P/D CUPTI short ROI와 2 GiB D collector buffer를 유지했다. Clean, profile, clean 각 200/200을 완료했고 E/P/D clock 및 collector coverage를 확보했다. 관측의 5% qualification은 이번에도 **FAIL**이었다. 아래 spread는 두 clean 회차의 차이, deviation은 profile에서 두 clean의 중앙값과의 상대 차이다.

| 측정 | Clean before | CUPTI/GPM | Clean after | Clean spread | Profile deviation |
|---|---:|---:|---:|---:|---:|
| TTFT p50 | 3.461 s | 3.497 s | 3.289 s | **5.09%** | 3.60% |
| TTFT p95 | 8.007 s | 7.433 s | 7.847 s | 2.01% | **6.22%** |
| ITL p99 | 12.829 ms | 12.474 ms | 13.084 ms | 1.97% | 3.72% |
| TPOT p95 | 5.209 ms | 6.032 ms | 5.337 ms | 2.42% | **14.40%** |
| 첫 도착부터 drain | 82.964 s | 83.745 s | 82.626 s | 0.41% | 1.15% |

| 두 Trace의 동일 경계 | 초기 | 관측 빈도 감소 |
|---|---:|---:|
| E/P/D kernel-busy | 70.10/83.38/50.58% | 70.12/83.68/49.93% |
| D FA3 attention 비중, 분모 D kernel-busy | 11.32% | 11.34% |
| D attention과 E 물리 idle의 교집합 | 30.29% | 27.90% |
| D attention과 E 물리 idle 및 HTTP 미처리 없음의 교집합 | 11.63% | 11.22% |
| D attention과 P 물리 idle 및 HTTP 미처리 없음의 교집합 | 5.24% | 4.86% |
| 500 µs 이상 E idle 및 HTTP 미처리 없음, 사후 fit | 13.70% | 13.08% |
| E DRAM BW utilization, D attention 동시 window의 시간가중 평균 | 21.93% | 21.21% |

두 물리 Trace는 **DIAGNOSTIC**이다. 빈도를 낮춘 회차에서 수집된 22개 Decode-only scheduler call은 실제 batch p50 1, p95 2.95, 최대 4였고, 합친 prompt rows p50 11,118, p95 30,093, 최대 37,135였다. 이 sample은 초기 95개보다 적다. 이 수치로 미래 encode 도착과 보조 연산의 안전한 admission 확률을 계산하려면 별도 검증이 필요하다. [낮은 관측 빈도의 원본 집계](../encoder_assisted_decode_attention/results/native_profile_low_observer.json) · [동일 경계의 GPU timeline](../encoder_assisted_decode_attention/figures/native_epd_timeline_low_observer.png) · [E/P/D 교집합](../encoder_assisted_decode_attention/figures/native_encoder_opportunity_low_observer.png) · [추가 물리 실행의 SHA-256 증빙](../encoder_assisted_decode_attention/results/native_extended_resource_ledger.json)

```bash
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.native_profile \
  --asset-repo "$asset_repo" --receipt "$asset_receipt" --run-dir ../h200_eada_runs/native_profile_low_observer \
  --mode native-strong --decode-cap free --qps 2.5 --profile --short-roi --profile-state-ms 1000 \
  --profile-decode-buffer-gib 2 --main100-cycles 2 --gpm-period 1.2 --defer-client-raw --wall-seconds 1700
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.motivation_analysis \
  --run-dir ../h200_eada_runs/native_profile_low_observer --tag native_strong_free_2p5_sustained_r2_profile \
  --out "$study/results/native_profile_low_observer.json" \
  --timeline-csv "$study/results/native_profile_low_observer_timeline.csv"
"$epd_python" -m poc_flashvep.encoder_assisted_decode_attention.native_resource_ledger \
  --raw-root ../h200_eada_runs --include-extended --out /tmp/eada-native-extended-resource-ledger-repro.json
```

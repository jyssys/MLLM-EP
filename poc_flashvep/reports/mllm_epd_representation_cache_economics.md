# MLLM 표현별 저장 비용과 H200 전송 비용

E GPU의 캐시 가치를 판단하려면, 재사용 가능한 표현 하나가 차지하는 공간과 적중 때 절약하는 연산을 따로 볼 필요가 있다. 이 실험은 원본 Qwen2.5-VL-7B 체크포인트의 네 Main100 요청을 사용했다. 요청은 실험 전 이미지 1장, 소수, 중간 문서, 긴 다중 페이지 문서를 포함하도록 고정했다. E는 Vision Encoder, P는 기존 언어 모델 Prefill, D는 전체 로컬 paged KV를 사용하는 FA3 Decode를 실행했다.

## 계산과 관측 범위

R0는 원본 압축 이미지 파일의 실제 byte 수다. R1은 Processor가 만든 pixel tensor의 FP32 저장량과 E에 입력된 BF16 저장량을 각각 기록했다. R2는 Visual Merger 직전 tensor의 실제 shape 및 dtype에서 계산했다. R3는 언어 모델에 전달되는 Merger 출력이다. R4는 전체 요청 Prefill 결과와 마지막 `<|vision_end|>`까지의 이미지 완결 Prefix를 실제 Forward해 얻은 28층 BF16 K/V 총량이다. 동일 prefix 여부와 vLLM APC의 사용 가능한 block-aligned 경계는 별도로 확인해야 한다. R5는 D에서 append하는 16개 생성 token에 해당하는 K/V 물리 형상이다. 각 요청의 Image Grid, Token IDs, 원래 MRoPE를 확인한 뒤 실행했다.

전송 시간은 실제 H200 GPU의 CUDA event로 측정했다. CPU 경로에는 pinned host memory, GPU 경로에는 NVLink peer DMA를 사용했다. 사전 할당한 연속 버퍼의 1, 16, 64, 256MiB, 1GiB와 실요청 R3 및 R4 길이를 사용했다. 저장 위치 조회, Pack/Unpack, NIXL handoff, Scheduler 대기는 이 물리 전송 시간에 포함되지 않는다. 이 값은 실제 캐시 load의 전송 부분에 대한 하한에 가깝다. 동시 실행은 별도 stream의 CUDA event enclosure가 양의 시간으로 겹친 표본을 집계했다. 같은 GPU의 같은 origin으로 stage와 copy event를 비교했다.

## 실제 표현 크기와 재계산 비용

같은 요청에서 이미지 완결 Prefix KV는 LM-ready Embedding의 **8.01–8.25배**였다. Embedding은 BF16 3,584차원으로 Visual Token당 **7,168B**였다. Full28 KV는 4개 KV head, head dimension 128, BF16 K/V를 합쳐 LM Token당 **57,344B**였다. D에서 실제로 생성한 16개 Token의 증분 KV는 **0.875MiB**였다. 아래 표는 전체 문서의 합계이며 per-image 값과 전체 요청 KV도 원본에 남겼다. [원본 측정 JSONL](../mllm_epd_memory_asymmetry_discovery/results/physical_economics.jsonl) · [표현 크기 Figure](../mllm_epd_memory_asymmetry_discovery/figures/representation_sizes.png)

| 요청 | 입력 Token | R0 원본 | R1 BF16 입력 | R2 Merger 직전 | R3 LM Embedding | R4 이미지 완결 Prefix KV |
|---|---:|---:|---:|---:|---:|---:|
| DocVQA · 1장 | 556 | 0.225MiB | 4.683MiB | 5.098MiB | 3.568MiB | 29.422MiB |
| Mantis-Eval · 2장 | 2,343 | 1.201MiB | 20.295MiB | 22.090MiB | 15.463MiB | 124.688MiB |
| MMLongBench-Doc · 21장 | 8,007 | 1.816MiB | 71.194MiB | 77.490MiB | 54.243MiB | 437.008MiB |
| MP-DocVQA · 6장 | 28,845 | 2.956MiB | 258.398MiB | 281.250MiB | 196.875MiB | 1,576.422MiB |

유효한 히트당 Prefix KV는 Encoder와 이미지 Prefix Prefill을 모두 절약할 수 있다. 같은 용량에서는 Embedding의 잠재 절약 시간이 Prefix KV의 **4.85–5.83배**였다. Embedding의 E→P 이동은 기존 EPD에서도 필요하므로 이 계산에서 공통 비용으로 처리했다. Prefix KV는 `E Encoding + 이미지 완결 P Prefill − E→P Prefix KV copy`로 계산했다. GPU stage 시간을 저장량으로 나눈 값이며, 실제 절약량은 적중률, 유효한 Prefix 경계, 관리 비용, Queueing에 달려 있다. [계산 CSV](../mllm_epd_memory_asymmetry_discovery/results/representation_utility.csv) · [재계산과 이동 Figure](../mllm_epd_memory_asymmetry_discovery/figures/recompute_vs_transfer.png) · [GiB당 절약 Figure](../mllm_epd_memory_asymmetry_discovery/figures/cache_utility_per_gib.png)

| 입력 Token | Encoding GPU | P Prefix GPU | E→P R3 | CPU→P R3 | E→P R4 | CPU→P R4 | R3 / R4 절약 시간 (ms/GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 556 | 43.62ms | 18.17ms | 0.017ms | 0.071ms | 0.090ms | 0.563ms | 12,519 / 2,147 |
| 2,343 | 146.79ms | 60.82ms | 0.052ms | 0.297ms | 0.344ms | 2.366ms | 9,721 / 1,702 |
| 8,007 | 506.85ms | 238.03ms | 0.157ms | 1.031ms | 1.168ms | 8.307ms | 9,568 / 1,743 |
| 28,845 | 1,825.68ms | 1,194.29ms | 0.536ms | 3.749ms | 4.173ms | 29.906ms | 9,496 / 1,959 |

동일 이미지 묶음을 같은 형태로 재실행한 Embedding은 **4/4 byte-identical**이었다. 실제 묶음에서 첫 또는 마지막 이미지를 꺼내 단독 인코딩한 대조는 **3/7 byte-identical**이었다. Mantis-Eval과 MMLongBench-Doc의 4개 비교에서 relative L2 **0.0569–0.0679**가 발생했다. 이 수치는 사용한 HF SDPA BF16 경로에 해당한다. 이미지별 독립 재사용을 적용하려면 실제 서비스의 Encoder batching과 처리 순서에서 다시 검증하고, 필요하면 동일 묶음·동일 실행 계약을 캐시 키에 넣어야 한다. Full28 KV 크기와 MRoPE 일치는 실제 tensor에서 확인했다. Prefix 복원 이후 Greedy와 Task Quality는 별도의 온라인 Correctness 실험 범위다. [표현별 Correctness 원본](../mllm_epd_memory_asymmetry_discovery/results/physical_economics_summary.json)

## 물리 전송, 메모리와 간섭

H200 GPU 4/5/6의 모든 E/P/D 쌍은 NV18 연결이다. 1GiB 연속 peer copy는 여섯 방향에서 **2.710–2.719ms**, pinned CPU copy는 여섯 방향에서 **19.431–19.464ms**였다. 기준 버퍼 5개, 실제 R3/R4 8개에 대해 12방향 3그룹을 측정했다. 표의 값은 그룹당 CUDA event 시간에서 copy 횟수를 나눈 뒤 중앙값을 취했다. [모든 전송 경로 CSV](../mllm_epd_memory_asymmetry_discovery/results/physical_transfer.csv) · [GPU와 원본 hash](../mllm_epd_memory_asymmetry_discovery/results/physical_lease_receipt.json)

독립 계측에서 E의 Vision Weight는 **1.260GiB**, P와 D의 Language Weight는 각각 **14.185GiB**였다. 전송 scratch를 생성하기 전 Torch reservation은 E/P/D **11.24 / 26.66 / 18.71GiB**였다. D에는 2.99GiB의 실험용 paged KV arena가 포함된다. 동일 하드웨어의 NativeStrong 용량 실험은 42GiB P/D KV 설정에서 수행했으며 활성 캐시의 용량과 SLO는 NativeStrong 관측 결과를 기준으로 평가했다. [독립 계측 메모리](../mllm_epd_memory_asymmetry_discovery/results/physical_economics_summary.json) · [NativeStrong 상주 메모리 실험](../mllm_epd_memory_asymmetry_discovery/results/native_residency_sweep.json)

8,007-token 요청의 실제 Encoder, Prefill, Full28 Local FA3 Decode와 전송을 각각 3회 짝지었다. 전송 burst마다 1GiB를 16MiB 또는 256MiB 단위로 나누어 24조건을 계측했고, 72/72 짝에서 stage와 transfer event enclosure가 양의 시간으로 겹쳤다. 16MiB P→E 전송과 E Encoding을 겹치면 표본 중앙값에서 copy enclosure **10.6%**, Encoding enclosure **5.5%** 증가했다. 같은 granularity와 경로의 P active 조건에서는 각각 **98.8%**, **6.1%** 증가했다. 16MiB E→D와 D active 조건의 copy enclosure는 **152.2%** 증가했다. stage enclosure의 음의 변화도 관측됐으며 3회 반복과 host dispatch를 포함한 구간 변화로 해석했다. 계측 범위는 CUDA event 구간이고 NVLink/HBM counter는 미측정이다. [간섭 CSV](../mllm_epd_memory_asymmetry_discovery/results/transfer_interference.csv) · [전송·연산 간섭 Figure](../mllm_epd_memory_asymmetry_discovery/figures/transfer_interference.png)

## 기존 NativeStrong의 대기 증거

이전 NativeStrong Free의 3GPU EPD를 그대로 실행한 QPS1, 2.5, 5의 300요청을 EADA 원본 proxy trace와 조인했다. Free 설정에서 proxy의 admission 대기 필드는 **0/300**이었다. Free 경로에는 해당 gate가 없어 이 값으로 D EngineCore 대기를 관측할 수 없다. P-ready→D-submit p95는 각 부하에서 **0.386, 0.429, 0.449ms**였다. 이 구간에 동시에 걸친 요청은 최대 1개였고 가장 큰 요청의 토큰 형상은 **1.634GiB**였다. 첫 SSE bytes까지 넓히면 동시에 최대 3개였고 토큰 형상은 부하별 **1.80–1.91GiB**였다. 이 구간에는 transport와 D 첫 실행이 포함된다. 토큰 형상과 P exporter가 실제로 KV를 해제한 시점은 서로 다른 관측값이다. 이전 worker 로그의 19개 표본에서 D KV 사용률은 최대 **2.2%**, P는 **5.4%**였다. D preemption은 기존 자료에 직접 집계 카운터가 없어 미측정이다. [기존 raw SHA와 집계](../mllm_epd_memory_asymmetry_discovery/results/historical_pressure.json) · [승인·압력 구간 계산](../mllm_epd_memory_asymmetry_discovery/historical_pressure.py)

Direction A에서 단일 P→D 대신 P→E→D가 필요하면 한 번의 peer copy가 추가된다. 이미 P에 보유 여유가 충분하고 위의 짧은 host admission 대기만 관측된 조건에서는 이를 회수할 수 있는 P backpressure 동기가 약하다. Direction D에는 별도의 실제 preemption 카운터와 resume 경로가 필요하다. 전송 비용만으로 preemption 빈도나 온라인 이득을 외삽하지 않는다.

| 입력 Token | P→D 직접 | P→E→D 두 구간 합 | D→E→D 두 구간 합 | D→CPU→D 두 구간 합 |
|---:|---:|---:|---:|---:|
| 556 | 0.089ms | 0.178ms | 0.178ms | 1.125ms |
| 2,343 | 0.338ms | 0.684ms | 0.684ms | 4.746ms |
| 8,007 | 1.169ms | 2.335ms | 2.336ms | 16.598ms |
| 28,845 | 4.171ms | 8.342ms | 8.341ms | 59.857ms |

각 합계는 동일한 R4 크기에서 두 격리 실측을 합친 물리 전송 하한이다. Request KV의 실제 여러 segment, 할당, 수명 보장, ACK, 복원 및 queue가 포함된 end-to-end 시간은 별도로 남는다. D→E→D 전송 우위는 preemption이 반복되는 이용 조건에서만 의미를 가질 수 있다. [물리 벤치마크 요약](../mllm_epd_memory_asymmetry_discovery/results/physical_economics_summary.json)

Direction E의 checkpoint copy에서는 실제 BF16 Language Weight 339개 tensor, **14.185GiB**를 D의 사전 할당 scratch로 옮겼다. E→D peer copy는 host wall 중앙값 **42.458ms**, 동일 payload의 Host→D는 64MiB pinned buffer 두 개를 이용한 staging으로 **560.564ms**였다. Host arm에는 pageable source에서 pinned buffer로 복사하는 시간도 포함했다. 마지막 Host→D 실행 뒤 Destination의 3개 표본 tensor는 원본과 일치했다. Multi-model 도착, Weight 설치, CUDA Graph 로딩, 모델 전환 중 E/P/D 지연은 후속 검증 범위다. [Weight copy 원본](../mllm_epd_memory_asymmetry_discovery/results/weight_copy.json) · [측정 코드](../mllm_epd_memory_asymmetry_discovery/weight_copy.py)

## 실행과 재현

아래 명령은 캠페인 GPU lease가 보장한 E/P/D 순서 안에서 실행한다. 공개 체크포인트와 보존한 Main100 자료를 기존 경로에서 읽는다. `physical_economics_lease/state/environment.json`에 실제 GPU UUID와 호출 인자를 남겼다.

```bash
runtime=../mllm_epd_runtime_env/bin/python
study=poc_flashvep.mllm_epd_memory_asymmetry_discovery
model=../mllm_epd_autosetup/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
manifest=../h200_discovery_runs/dataset/grouped_v2/discovery200.json
raw=../h200_memory_asymmetry_runs
"$runtime" -m "$study.physical_economics" --model "$model" --manifest "$manifest" --out "$raw/physical_economics.jsonl"
"$runtime" -m "$study.weight_copy" --model "$model" --destination 2 --out "$raw/weight_copy.json"
matplotlib_python=/home1/irteam/ddn/envs/kvp_tf453/bin/python
"$matplotlib_python" -m "$study.physical_economics_summary" "$raw/physical_economics.jsonl" \
  --out /tmp/mllm-physical-economics-repro.json --csv-dir /tmp/mllm-physical-economics-csv \
  --figure-dir /tmp/mllm-physical-economics-figures
"$runtime" -m "$study.historical_pressure" ../h200_eada_runs/native_scan_retry --out /tmp/mllm-historical-pressure.json
```

[측정 코드](../mllm_epd_memory_asymmetry_discovery/physical_economics.py) · [집계와 Figure 코드](../mllm_epd_memory_asymmetry_discovery/physical_economics_summary.py) · [EADA 종합 결과](https://github.com/jyssys/MLLM-EP/blob/07892cbe7193366ec422f1df82006c6e69f91cec/poc_flashvep/reports/mllm_encoder_assisted_decode_attention_final_summary.md)

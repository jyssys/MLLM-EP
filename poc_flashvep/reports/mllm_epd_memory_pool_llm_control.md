# 일반 LLM PD의 Prefix KV 계층 대조

일반 LLM에서도 다른 GPU를 Exact Prefix KV 저장소로 쓸 수 있는지 검증하기 위해 Qwen2.5-7B-Instruct BF16의 Native vLLM PD를 구성했다. Qwen2.5-VL-7B와 28 layers, 28 query heads, 4 KV heads, head dimension 128, BF16 KV를 맞췄다. 두 모델 모두 Full28 K/V에서 Token당 57,344B를 사용한다. Text의 원래 RoPE와 VL의 원래 MRoPE는 각 실행 경로에서 유지한다. [공식 Text 체크포인트](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28)

## 입력, 메모리와 실행 계약

고정 LongBench128의 Qasper, HotpotQA, NarrativeQA, GovReport에서 태스크마다 원본 Prompt 16개를 Seed로 골랐다. 64개 Prompt 뒤에 같은 64개를 요청 ID만 달리해 역순으로 재생한다. 64개 고유 문항의 입력은 2,527–28,173 Token이고 중앙값은 10,296.5 Token이다. 최근 저장된 문항을 먼저 다시 요청해 제한된 Tier에서의 실제 복원 경로를 노출하도록 Trace를 동결했다. 이 두 번째 구간은 제어된 Exact Reuse다. Prompt SHA와 Token 수는 각각 CPU와 실제 API에서 검증한다. 강제 Output은 16개 Token이다. [공식 LongBench 동결 Revision](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672) · [동결 Roster 및 모델 지문](../mllm_epd_memory_asymmetry_discovery/results/llm_pd_roster.json)

H200 4/5/6에 2P/1D를 구성했다. Prefix가 같은 요청은 언제나 같은 Prefill Worker로 보낸다. Token 수로 균형을 맞춘 두 P의 최초 64요청 KV Working Set은 Block128 반올림 기준 **18.252GiB / 18.238GiB**였다. 별도 1P/2D 대조에서는 P 한 대에 같은 요청을 보낸 뒤 두 D에 고정 shard를 사용한다. 세 GPU는 NV18 topology이며 모든 Active Decode는 각 D의 로컬 KV를 사용한다. P→D는 기존 NIXL handoff로 처리한다.

| 구성 | P의 Native KV | P별 별도 Prefix Tier | 목적 |
|---|---:|---:|---|
| 2P/1D Local42 | 42GiB | 0 | 일반적인 Native APC |
| 2P/1D Local24 | 24GiB | 0 | Tier 구성과 같은 P별 Cache Byte |
| 2P/1D CPU16 | 8GiB | 16GiB pinned host | 일반 GPU/CPU Tier |
| 2P/1D Peer16 | 8GiB | 16GiB 반대편 P GPU | 같은 GPU 예산의 Exact Peer Tier |
| 1P/2D Local42 | 42GiB | 0 | GPU 역할 재배분 |

모든 D는 42GiB KV와 APC를 사용한다. Peer16에서는 P0의 보조 저장소를 P1에, P1의 보조 저장소를 P0에 배치한다. 일반 vLLM OffloadingConnector의 해시, Block Page, Admission, Eviction을 그대로 쓴다. Peer Tier는 동일 메커니즘의 저장 위치를 CUDA peer memory로 바꾸며, 재사용 시 자신의 P GPU에 필요한 블록을 완전히 복원한 뒤 계산한다. 이 실험은 평균 Load에서 서로 다른 두 P에 저장소가 공존할 때의 대조를 제공한다. P-local KV를 확대했을 때의 이득도 같은 GPU 수에서 확인한다. [실행과 Proxy](../mllm_epd_memory_asymmetry_discovery/llm_pd_control.py) · [Peer OffloadingConnector](../mllm_epd_memory_asymmetry_discovery/peer_kv.py)

## 실측과 해석 범위

128건의 Open-loop cohort를 동일한 QPS2.5 Trace로 각 Arm에서 한 번씩 독립 시작했다. TTFT는 예정 Arrival부터 첫 생성 Token의 SSE까지, TPOT와 ITL은 실제 SSE Token ID를 기준으로 측정했다. SLO는 TTFT 최대 5,000ms, 모든 Client ITL 최대 34ms, TPOT 최대 33ms다. 고정 Output 실험에서는 Token fidelity를 비교했고 Natural Task Quality는 미실행이다.

P의 local APC hit와 external hit를 Prometheus의 Token counter에서 분리한다. vLLM의 `CPU_to_GPU`/`GPU_to_CPU` label은 기존 offload 인터페이스 방향 표시다. Peer16의 실제 장치와 할당량은 CUDA worker receipt를 기준으로 확인한다. NIXL의 D-side External Prefix counter에는 원래 P→D 이동도 포함되므로 P-tier 적중을 판단할 때 사용하지 않는다. P Cache를 가진 GPU Weight 차이만으로 Online Benefit을 외삽하지 않는다.

### Native Online 성능

다섯 Arm에서 **640/640**이 정확히 16개 Output Token으로 완료됐고 모든 요청이 Joint SLO를 충족했다. 모든 Worker의 Preemption counter 변화는 **0**이었다. Goodput은 약 47.45초의 한 Trace에서 완료된 SLO 요청 수이며, 지속 처리 용량은 별도 부하 검증이 필요하다. 동일 Cache Budget의 Local24가 두 Tier보다 낮은 TTFT p95를 기록했다. Practical 2P/1D Local42는 Local24와 같은 **719,616** P-local hit tokens를 기록했다. 1P/2D Local42는 ITL p99가 가장 낮았고 TTFT p95는 2P/1D Local42보다 **21.9%** 높았다. [전체 Online 집계](../mllm_epd_memory_asymmetry_discovery/results/llm_pd_online.json) · [전체 640개 요청 Timing](../mllm_epd_memory_asymmetry_discovery/results/llm_pd_request_timings.jsonl) · [Online Figure](../mllm_epd_memory_asymmetry_discovery/figures/llm_pd_online.png)

| 구성 | TTFT p95 | TPOT p95 | ITL p99 | E2E p95 | Joint SLO-goodput |
|---|---:|---:|---:|---:|---:|
| 2P/1D Local24 | 810.54ms | 4.46ms | 10.38ms | 860.93ms | 2.698 req/s |
| 2P/1D CPU16 | 823.24ms | 4.62ms | 11.04ms | 872.70ms | 2.696 req/s |
| 2P/1D Peer16 | 818.57ms | 4.75ms | 11.14ms | 883.08ms | 2.696 req/s |
| 2P/1D Local42 | 793.05ms | 4.59ms | 11.72ms | 842.05ms | 2.697 req/s |
| 1P/2D Local42 | 966.79ms | 4.37ms | 9.04ms | 1,021.25ms | 2.697 req/s |

2P/1D의 P-local 42GiB 설정에서 P GPU의 요청 종료 후 물리 여유는 각각 **78.44GiB**였다. 별도 NativeStrong Image EPD의 E GPU에서는 Working State에서 **98.83GiB**가 남았다. Text의 P는 **14.185GiB** LM Weight와 Active KV를 유지하고, Image E는 **1.260GiB** Encoder Weight와 Vision Runtime State를 유지했다. 비교는 관측된 Worker 메모리 상태를 보여준다. 각 Worker의 안전한 추가 저장량에는 최대 활성화와 실제 삽입·읽기 부하를 포함한 검증이 필요하다. [Text Worker 메모리](../mllm_epd_memory_asymmetry_discovery/results/llm_pd_online.json) · [NativeStrong Image EPD 메모리](../mllm_epd_memory_asymmetry_discovery/results/native_residency_sweep.json)

### 실제 적중과 이동 비용

Peer16은 양쪽 P에 각각 **15.996GiB**를 물리 할당했다. P 두 대 합계에서 **343,040 external hit tokens**와 **18.320GiB**의 실제 load를 기록했다. Host16은 **343,296 external hit tokens**, **18.334GiB** load였다. Load CUDA event 누적 합은 Peer **0.233s**, Host **0.503s**였다. 누적 합에는 여러 worker와 요청의 동시 실행이 포함된다. Peer의 두 번째 구간 TTFT p95는 Host보다 **17.9%** 낮았고 동일 Cache Budget의 Local24보다 **27.0%** 높았다. 기존 OffloadingConnector의 저장 장치를 GPU peer로 바꿨을 때 Text PD에서도 물리적 읽기와 CPU tier 대비 두 번째 구간 지연 절감이 재현됐다.

| 구성 | P-local hit tokens | P-tier hit tokens | 누적 tier load | 두 번째 구간 P HTTP p95 | 두 번째 구간 TTFT p95 |
|---|---:|---:|---:|---:|---:|
| Local24 | 719,616 | 0 | 0GiB | 151.65ms | 290.84ms |
| CPU16 | 337,792 | 343,296 | 18.334GiB | 351.97ms | 449.85ms |
| Peer16 | 337,920 | 343,040 | 18.320GiB | 221.66ms | 369.23ms |

동일한 Local24 안에서도 처음 요청과 반복 요청의 16개 Greedy Token 전체가 **53/64**에서 일치했다. Local24와 Peer16의 동일 Arrival 위치 간에는 **124/128**, CPU16은 **123/128**, Local42는 **126/128**, 1P/2D는 **125/128**이 일치했다. 캐시 온도, 실제 Batch, 수치 경로가 함께 달라지는 관측이므로 Tensor 정합성과 Task Quality는 별도 검증이 필요하다. 이 Text 대조에서는 Full-output Fidelity를 측정했다.

### Text PD와 Image EPD에서의 상대 효과

동일한 Cache Byte의 Local APC를 각자의 분모로 삼았다. Text PD는 P 2대 각각 `8GiB + 상대 P 16GiB`와 `Local24GiB`를 비교했고, Image EPD는 `P8GiB + E64GiB`와 `P-local72GiB`를 비교했다. 둘 다 QPS2.5와 고정 16개 Output을 사용했다. 문항, Prefix 길이, Cache Working Set, GPU 역할은 각 Trace의 원래 조건을 유지했다. Text에서 Peer는 TTFT p95 **+1.0%**, TPOT p95 **+6.5%**, ITL p99 **+7.3%**, Goodput **−0.05%**였다. Image에서는 각각 **+10.0%**, **−24.0%**, **−24.4%**, **−5.2%**였다. 단일 시작에서의 상대 변화이며, GPU 저장소의 공통 메커니즘과 각 Workload의 효과를 보여준다. 독립적인 MLLM-specific 성능 기여를 판단하려면 동일 용량의 Local APC를 넘는 온라인 이득과 표현별 재사용 효과가 필요하다. [Text PD와 Image EPD 상대 효과 Figure](../mllm_epd_memory_asymmetry_discovery/figures/llm_pd_vs_mllm_epd_peer.png)

## 재현

기존 핀 Runtime을 사용하고 Arm마다 다른 외부 Run Directory를 지정한다. 아래 예시는 Same-Cache-Budget의 기준 Arm이다. `--tier cpu --p-kv-gib 8 --tier-gib 16`, `--tier peer --p-kv-gib 8 --tier-gib 16`, `--p-kv-gib 42`로 나머지 2P/1D Arm을 실행한다. GPU 역할 비교에는 `--layout 1p2d --p-kv-gib 42`를 사용한다. `--freeze-only`는 GPU Lease와 서버를 열기 전에 Prompt, Token 및 Shard 배치를 점검한다.

```bash
runtime=../mllm_epd_runtime_env/bin/python
model=../h200_eada_runs/models/qwen25_7b
manifest=../h200_eada_runs/longbench/qwen_128.json
asset=../MLLM-EP-mllm-epd
"$runtime" -m poc_flashvep.mllm_epd_memory_asymmetry_discovery.llm_pd_control \
  --model "$model" --manifest "$manifest" --asset-repo "$asset" \
  --run-dir ../h200_memory_asymmetry_runs/llm_control_local24_repro --p-kv-gib 24
```

아래 명령으로 완료된 Run의 집계와 상대 효과 Figure를 재생성한다. EPD 입력은 원본 Summary와 같은 위치의 Run Directory에서 RAW SHA, Arrival 순서, 요청 ID를 검증한다. [집계 코드](../mllm_epd_memory_asymmetry_discovery/analyze_llm_pd_control.py)

```bash
runs=../h200_memory_asymmetry_runs
plot_python=/home1/irteam/ddn/envs/kvp_tf453/bin/python
MPLBACKEND=Agg "$plot_python" -m poc_flashvep.mllm_epd_memory_asymmetry_discovery.analyze_llm_pd_control \
  --arm "Local24=$runs/llm_2p1d_local24_n1" --arm "Peer16=$runs/llm_2p1d_peer16_p8_n1" \
  --arm "CPU16=$runs/llm_2p1d_host16_p8_n1" --arm "Local42=$runs/llm_2p1d_local42_n1" \
  --arm "1P2D Local42=$runs/llm_1p2d_local42_n1" --reference Local24 --out /tmp/llm-pd-online.json \
  --csv /tmp/llm-pd-online.csv --figure /tmp/llm-pd-online.png --timings /tmp/llm-pd-timings.jsonl \
  --epd-comparison "$runs/w5_storage_screen.json" pressure188_local72_n1 pressure188_peer64_p8_n1 Peer16 \
  --cross-figure /tmp/llm-pd-vs-mllm-epd.png
```

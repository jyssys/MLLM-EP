# H200 Runtime 절대 성능 Sanity

## 비교 계약

동결 Calibration32의 동일 요청·이미지·순서·실제 Processor와 모델 Revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`를 쓴다. 32건의 Manifest Input 합계는 374,882 tokens, Visual 합계는 372,749 rows, 이미지는 405장이다. Input은 평균 11,715.06, 중앙값 10,754, 543–30,195 tokens다. 한 번 Warm-up 이후 한 요청씩 Natural EOS, Greedy, Cap2048을 수행한다. 실제 출력 수가 달라지면 각 Arm의 Output을 함께 기록한다. 절대 Latency와 Runtime 계층의 진단이 목적이며 두 Arm의 GPU 예산이 서로 다르다. Main100 온라인 QPS 결과와 합치지 않는다. [Setup 환경 및 정합성](mllm_epd_h200_environment_and_correctness.md)

Stock은 같은 공식 vLLM Wheel 위에서 Aggregated 단일 H200, Native는 실제 Vision Encode·CUDA IPC Embedding·NIXL P→D를 사용하는 H200 세 대다. NativeStrong의 Stock LLM Compiler와 원래 E 설정을 유지한다. 단일 GPU Stock은 기본 vLLM Processor/Renderer 및 1GB MM Cache를 사용하는 반면 Native는 분리한 E의 Renderer2·MM Cache0·Sync Scheduling을 사용한다. 이 차이는 각 실행 경로의 일부다. P/D B8192·S1024·KV42GiB·Block128·Prefix Cache Off를 보존한다. Runtime Compiler·CUDA Graph·Batch·warmup·CPU/NUMA·버너 상태는 각 Run의 로그에 남긴다. 동일 요청의 로컬 절대 비교로 Stock 단일 GPU와 EPD 경로를 확인하고 TTFT 분해는 Native의 HTTP·Client 경계로 해석한다.

## 외부 수치의 기준과 적용 범위

공식 vLLM Recipe의 7B VisionArena-Chat 128건 예시는 13.09 req/s, Output 1,455.11 tok/s, TTFT 중앙값 4,751.45ms, ITL p99 653.85ms다. 7B Synthetic 입력/출력 길이 8000/1000, 128건 예시는 2.83 req/s, Output 2,581.01 tok/s, TTFT 중앙값 10,560.30ms, ITL p99 196.22ms다. 해당 수치는 Recipe의 예시 결과이며 동일 Calibration32·Serial·H200·GPU 예산의 비교값은 제공하지 않는다. 입력·출력 길이, 부하와 병렬 배치가 달라 수치 차이를 Runtime 이상으로 해석하지 않는다. [공식 Recipe](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen2.5-VL.html#benchmarking)

DuetServe Fig. 2는 텍스트 Qwen3-8B 8000/200에서 H100 2대 위의 Agg-vLLM와 1P1D를 비교했다. 분리형의 QPS 증가에 따른 TBT 안정과 TTFT·처리량 변화는 부하 Sweep 선택에 참고하고, Calibration32의 절대 수치 기준은 이 서버의 같은 요청 Stock/Native에서 얻는다. [DuetServe Fig. 2](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)

<!-- WIDE_SANITY_START -->

| Arm | Evidence | Completions | Input / output tokens | TTFT p50 / p95 ms | ITL p99 ms | TPOT p95 ms | Requests/s | Output tokens/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stock vLLM 1 GPU | MEASURED_ONLINE | 32 | 374882 / 1152 | 866.02 / 2226.21 | 5.50 | 4.77 | 0.89 | 32.11 |
| NativeStrong EPD 3 GPU | MEASURED_ONLINE | 32 | 374882 / 1207 | 882.56 / 2175.52 | 5.39 | 4.70 | 0.84 | 31.54 |

| Arm | Matched proxy rows | E HTTP p50 ms | P HTTP p50 ms | D HTTP p50 ms |
| --- | --- | --- | --- | --- |
| NativeStrong EPD 3 GPU | 32 | 381.80 | 413.11 | 132.62 |

The separately addressed E/P/D workers expose these host phases. Stock has a single HTTP serving worker. The phase medians include queues, transfers, and HTTP work.

Both arms completed the same ordered 32 requests with matching prompt and payload hashes, engine input token IDs, runtime receipt, wheel hash, actual input lengths, and seed. The EPD minus stock TTFT difference was +16.53 ms at p50 and -50.69 ms at p95.

Natural EOS output counts agreed on 29/32 requests. Exact output token IDs agreed on 29/32. The matched calibration cohort does not establish held-out task quality.

| Request | Stock / Native output tokens | First divergent output position |
| --- | --- | --- |
| TIGER-Lab__Mantis-Eval__wc_17 | 187 / 183 | 89 |
| lmms-lab__MP-DocVQA__60933 | 16 / 58 | 3 |
| yubo2333__MMLongBench-Doc__975 | 258 / 275 | 71 |

<!-- WIDE_SANITY_END -->

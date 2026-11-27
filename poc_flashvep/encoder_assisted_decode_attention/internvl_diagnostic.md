# InternVL3-8B 실행 및 실제 KV 분할 진단

OpenGVLab의 Apache-2.0 InternVL3-8B `853e3a797a661694b1b8ece0cb72dc2b23e3dac9`를 설치된 vLLM 0.26의 `InternVLChatModel`로 로드했다. Vision Encoder와 Projector는 Qwen2.5-VL과 다르다. Language Backbone은 Qwen2 계열이고 28 Layers, 28 Query Heads, 4 KV Heads, Head Dimension 128을 공유한다. 로컬 Config의 SHA를 검증했으며 모델 실행에는 vLLM 기본 구현을 사용했다. [Checkpoint](https://huggingface.co/OpenGVLab/InternVL3-8B/tree/853e3a797a661694b1b8ece0cb72dc2b23e3dac9) · [다운로드 검증](results/internvl3_8b_checkpoint.json) · [환경 및 실행 기록](results/internvl3_diagnostic/fingerprint.json)

## 단일 GPU Native 실행

H200 GPU6에서 원본 Processor와 이미지를 사용해 Main100의 서로 다른 네 요청을 각각 세 번 실행했다. Greedy, BF16, 고정 출력 16, Eager FA3, 8 GiB KV Pool, Prefix Cache Off 조건이다. 8-image 요청이 Main100에 없어 사전 정의 거리 규칙으로 6-image를 선택했다. 모든 요청이 완료됐다. 표의 TTFT와 TPOT는 각 요청의 vLLM 내부 기록 중앙값이다. TPOT는 마지막 토큰과 첫 토큰 사이 시간을 15로 나눴다. [요청별 원시 기록](results/internvl3_diagnostic/native.json)

| 요청 | 이미지 | Input Tokens | Visual Tokens | TTFT ms | TPOT ms | Peak Torch Allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| DocVQA 15382 | 1 | 3,385 | 3,328 | 203.75 | 7.55 | 23.50 |
| Mantis m_17 | 3 | 5,952 | 5,888 | 295.49 | 7.56 | 23.82 |
| MP-DocVQA 44820 | 6 | 20,040 | 19,968 | 1,317.79 | 7.76 | 23.92 |
| MMLongBench-Doc 984 | 16 | 28,784 | 28,672 | 1,668.69 | 8.14 | 23.98 |

GPU에 상주한 Vision Encoder 파라미터는 **0.566 GiB**, Projector는 **0.051 GiB**, Language Model은 **14.180 GiB**였다. 16-image에서 Torch Peak Reserved는 26.85 GiB였다. 메모리 수치는 통합 모델과 지정 KV Pool이 한 GPU에 있는 환경이다. InternVL의 E-only Safe KV Capacity는 별도 측정이 필요하다. [모델별 실제 Allocated Bytes](results/internvl3_diagnostic/native.json)

## 두 GPU의 실제 InternVL Attention

16-image 요청 첫 Decode의 Layer 0에서 Native post-RoPE Query와 KV를 수집했다. Prompt 28,784행과 첫 Decode 입력 1행을 사용했다. Query는 `[1, 28, 128]`, K/V는 각각 `[28785, 4, 128]` BF16이었다. 별도 Lease의 D6/E4 NV18에서 같은 Q/K/V를 Local FA3와 두 장치 Graph의 물리적 Peer Transfer, 부분 FA3, LSE Merge에 사용했다. 이 Lease에서 E4는 KV와 Attention만 담당했다. [Operator Raw JSONL](results/internvl3_diagnostic/operator.jsonl) · [Q/K/V 지문](results/internvl3_diagnostic/native.json)

| 분할 방식 | E KV 행 | Local Graph µs | Split Graph µs |
|---|---:|---:|---:|
| Generic prefix 50% | 14,392 | 25.96 | 40.98 |
| 동일 행 수 Generic prefix | 14,336 | 25.18 | 40.85 |
| Visual-first 50% | 14,336 | 25.76 | 41.38 |

전체 12개 Arm에서 Split Graph는 각 Local Graph보다 **51.0–92.3% 느렸다**. 같은 행 수에서 Visual-first의 Q/K/V 선택 이득은 비율 전반에서 일관되지 않았다. Arm마다 64회 반복을 묶은 CUDA Event 측정을 세 번 했다. Local baseline은 `[0, 1, 2, 4, 8]` Split-K 후보 중 실측 중앙값으로 선택했다. 최대 Graph 출력 relative L2는 **0.00260**, Eager의 최대 LSE 오차는 **6.10e-5**였다. Layer 0 논리적 왕복 데이터는 Query 7,168 + Partial Output 7,168 + LSE 112로 **14,448 B**였다. 이 비교에서 별도의 NVLink 바이트 카운터를 수집하지 않았다. [Operator Raw JSONL](results/internvl3_diagnostic/operator.jsonl)

**Evidence level** `ISOLATED_GPU_MEASUREMENT` 및 `CORRECTNESS_ONLY`. InternVL Full28 EADA, Encoder와 동시 실행하는 두 번째 모델의 간섭, Open-loop EPD 이득 및 공식 Task Quality는 `NOT_RUN`이다. 실제 모델의 이 단일 Layer 결과로 Gate 3을 통과할 수 없다.

## 재현

연구 Worktree에서 다음 명령으로 D6 단일 GPU 출력과 D6/E4의 독립 Operator 비교를 순서대로 실행한다. 큰 Q/K/V Fixture는 로컬 결과 디렉터리에 둔다. [실행 스크립트](run_internvl.sh) · [소스 지문](results/internvl3_diagnostic/source.sha256)

```bash
bash poc_flashvep/encoder_assisted_decode_attention/run_internvl.sh \
  ../mllm_epd_runtime_env ../h200_eada_runs 6 4 internvl3_diag_repeat
```

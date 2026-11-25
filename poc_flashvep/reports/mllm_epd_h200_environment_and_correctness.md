# H200 MLLM EPD 환경과 인수인계 정합성

2026-11-25 H200에서 기존 Setup Receipt와 N1/N8 실측을 확인했다. 원본 모델·Main100·Calibration32·NPZ를 현지에서 읽어 해시를 재검증했다. 모델과 이미지를 다시 다운로드하거나 Dataset을 재선정하지 않았다. NativeCommon과 Token10의 이전 Setup 결과는 상태 정합성의 근거다. Native N8은 2개 요청의 Greedy Output이 달랐고 전체 Task Quality는 `PENDING`이다. Batch1 Admission은 이번 캠페인의 변경 사항이며 별도의 정합성·Prepared Batch 근거가 필요하다.

## Checkout과 환경

기존 `MLLM-EP-mllm-epd`는 `MLLM-EP-mllm-epd.filtered`의 Alias다. Sparse/partial/shallow Clone에서 실제 Branch `flashvep/mllm-epd-prefill-heavy-token-split-20261124`와 HEAD `449c3e1873c65206435a6bb9411a889a62e55be7`를 확인했다. 복원된 미추적 Media가 존재하며 기존 Worktree의 Tracked Diff는 없다. 새 연구는 같은 Base의 별도 `h200-research` Worktree와 `flashvep/mllm-epd-h200-prefill-heavy-batch-matched` 브랜치를 사용한다. [인수인계 계약](../mllm_epd_handoff/README.md) · [빠른 설정](../mllm_epd_handoff/quick_setup_prompt_ko.md) · [재개 계약](../mllm_epd_handoff/resume_prompt_ko.md)

| 항목 | 이전 H100 | 현재 H200 |
|---|---|---|
| GPU | H100 HBM3 8대, 81,559MiB/대 | H200 8대, `nvidia-smi` 143,771MiB/대, 실제 Worker `total_memory` 150,109,880,320B |
| NVIDIA Driver | 570.211.01 | 580.126.16, `nvidia-smi` CUDA 13.0, 실험 CUDA Toolkit 12.9.86 |
| Python | 3.14.6, Anaconda, GCC 14.3.0 | 3.14.6, conda-forge, GCC 14.4.0 |
| Core Packages | vLLM 0.26.0+cu129, Torch 2.11.0+cu129, Transformers 5.16.1, NIXL 1.4.1 | 동일 버전 문자열, TorchVision 0.26.0+cu129, NIXL CUDA12/13 배포판 1.4.1, TorchCodec 0.16.0+cu129 |
| vLLM Wheel SHA256 | `7632856147650da3ed8d1652b1b05ffaadcc62ea8e910fdaa6f8ce055b201ebf` | `6ce4ca30616f0a35810391015622b197a7b8b267ed27f8716f0789db79ff578b`, 설치 `direct_url.json` 확인 |
| OS Kernel | 5.15.0-174-generic, glibc 2.35 | 6.8.0-71-generic |

엔진 로그에서 V1 Runner, `FLASH_ATTN`/FlashAttention 3의 Vision·LM Attention, FlashInfer Sampling, Inductor Compile, `FULL_AND_PIECEWISE` CUDA Graph Mode와 Capture Size 1부터 시작하는 설정을 확인했다. `CUDAGraphWrapper`는 논리 Prepared Batch와 실제 Graph Padding이 다를 수 있음을 뜻한다. FlashInfer 0.6.14, Triton 3.6.0, CUPTI 12.9.79가 설치돼 있다. 기존 Setup은 CUDA 13용 TorchCodec 대신 CUDA 12.9용 Wheel을 적용하고 CUDA Toolkit과 필요한 FlashInfer JIT 헤더를 연결해 기동했다. 같은 버전 문자열을 동일 Binary로 취급하지 않는다. [H100 환경](../mllm_epd_handoff/source_environment.json)

| H200 인덱스 | GPU UUID | PCI Bus | NUMA |
|---:|---|---|---:|
| 0 | `GPU-6b56b49e-2472-9e55-7d0a-74a087b4132e` | `19:00.0` | 0 |
| 1 | `GPU-4f143625-73b8-2372-5e1a-0bf13de11d61` | `3B:00.0` | 0 |
| 2 | `GPU-9d1efd41-3a8a-196e-690e-4e0463a802e9` | `4C:00.0` | 0 |
| 3 | `GPU-37163db1-5a6b-5702-e7dd-7791d5cb601b` | `5D:00.0` | 0 |
| 4 | `GPU-7cdfd11a-6f4d-3021-186b-5cd6b14dfd1e` | `9B:00.0` | 1 |
| 5 | `GPU-5b4a9509-9fe6-9390-fb6e-1990904e7fd9` | `BB:00.0` | 1 |
| 6 | `GPU-ba1fe988-9759-1ffd-00f5-854a961698ce` | `CB:00.0` | 1 |
| 7 | `GPU-20006e9b-3b48-7c54-cdb2-c503bde72bfc` | `DB:00.0` | 1 |

`nvidia-smi topo -m`에서 모든 GPU 쌍이 NV18로 연결됐다. Intel Xeon Platinum 8462Y+ 2소켓, 128 논리 CPU, NUMA 0/1 구성을 확인했다. 워크스페이스 Lustre 80T 중 약 57T 가용, `/dev/shm` 1001G다. 최초 8대 모두 0 MiB·0%였고, 이어 같은 계정의 승인된 `train_lm.py` Burner가 각 GPU에 올라온 것을 확인했다. 역사적 PID는 새 실험 소유권으로 쓰지 않는다. 본 캠페인 GPU/CPU 프로세스 식별과 장치 선택은 각 Run의 Lease를 따른다.

## 데이터와 모델 검증

Setup Receipt `download_ready_1795600931728841878.json`의 SHA256은 `7f5a6f419e8540db11f923fe0815e57aad8cda7d9fd3b8483914de1f317f6fbe`다. `download_plan.json`의 pinned 모델 파일 15개, 총 16,595,979,762B를 해시와 크기로 확인했다. Main100+Calibration32의 이미지 1,530개와 NPZ 132개, 원본 전체 NPZ 432개 및 동결 Manifest 3개를 별도로 해시 검증했다. 각 파생 Manifest의 `images[].path`와 `token_metadata_file`을 원본 Locator로 복구해 순서와 전체 JSON 구조가 같음을 확인했다. 원본 PDF/Parquet Locator는 Provenance로 그대로 유지된다. [고정 다운로드 계획](../mllm_epd_handoff/download_plan.json)

| 자료 | 요청 수 | 원본 SHA256 | 파생 SHA256 |
|---|---:|---|---|
| Main100 | 100 | `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1` | `1d7221b2391c5239ca0a85a109d3b93d46ab1a03c4a42f4caaf44d0a9a437151` |
| Calibration32 | 32 | `0e6409b167e80849dcbdd91df5f185bd79e235c562eab4ebfc885618caacfa0d` | `5871389d82291f54a8342a19cc1ede7a6b4acf1b88e6891c28b80d067714a979` |

## 기존 H200 정합성 재검토

2026-11-25 19:10–19:17 KST의 `mllm_epd_autosetup/correctness_20261125_retry3`에서 GPU4/E, GPU5/P, GPU6/D로 검사했다. B8192/S1024 및 42GiB는 P/D Worker에서 확인했다. E Worker는 B114688, Renderer2, MM Cache0, Sync Scheduling, 원래 Vision만 상주한다. NativeCommon은 P/D 공통 Reduction 설정을 사용했다. `N8`은 요청 8개를 검사한 이름이며 Actual Decode Batch8을 뜻하지 않는다.

| Path | 상태 정합성 N1/N8 | Strict Greedy N1/N8 | 근거 |
|---|---|---|---|
| NativeCommon | 1/1, 8/8 | 1/1, 6/8 | Native import L−1행과 마지막 Query 재계산을 따로 검사 |
| Token10 | 1/1, 8/8 | 1/1, 8/8 | 같은 원래 E, 실제 Prefix KV 전달·D Suffix 및 Natural Full Decode |

Native에서 `yubo2333__MMLongBench-Doc__975`의 Teacher/Native 출력 길이는 286/259, `yubo2333__MMLongBench-Doc__688`은 56/60이다. 두 요청 모두 상태 Gate는 통과했다. 다른 Geometry의 Greedy 차이를 그대로 남겼으며 원인을 이 검사의 Token Split 오류나 Task Quality 변화로 확정하지 않았다. 이전 H100에서 동일 이미지의 2-image→5-image Encoder Batch 비교에도 8.5–10.4% Embedding Relative L2 변화가 있었다. N1/N8 결과는 `CORRECTNESS_PASS` 범위를 지지한다. H200 전체 Task Quality, NativeStrong Stock의 독립 수치 계약, B1 변경 후 동작과 Actual Scheduled/Prepared Decode Batch는 별도 근거가 필요하다. [이전 H100 정합성 보고서](mllm_epd_generic_token_split_correctness.md)

기존 Setup의 `analysis/Native.json` SHA256은 `4da9f78887e3067f521cf053e383fe3ec5c139491ed91b649478b9ec889afa6b`, `analysis/Token10.json`은 `f01be4a0fa61755fde348af4191e710643df48ba2ebda742d6a99ec669d40a3c`, `state/environment.json`은 `9d38964383be0e79d78b658dd34e3da95d56127a493df9a42637ef2e965135c3`다. 원본 Raw와 Setup은 같은 서버에 보존했다. 이번 연구의 B1 정책과 Phase A 온라인 결과는 새 Namespace에서 추적한다.

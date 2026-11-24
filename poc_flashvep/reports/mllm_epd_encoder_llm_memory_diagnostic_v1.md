# E GPU와 추가 LLM의 메모리 적재 검사 — C16

판정은 `OBSERVED_CO_RESIDENCY_FIT`다. 증거 수준은 `ISOLATED_MEMORY_MEASUREMENT`다.
GPU4의 H100 SXM 80GB 한 대를 사용했다. 독립 초기화는 1회다.
모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
Original BF16 Vision Encoder와 기존 Main100 원문을 유지했다.

E는 다음 실제 입력을 추가 모델 적재 전후에 각각 처리했다.
선정 규칙은 Main100의 최대 Input, 최대 이미지 수, 최대 단일 이미지 Visual Tokens였다.
성능을 확인한 뒤 입력을 교체하지 않았다.

| Request | Input Tokens | Images |
|---|---:|---:|
| MP-DocVQA 58411 | 30536 | 6 |
| MMLongBench-Doc 723 | 24450 | 60 |
| MP-DocVQA 65398 | 15527 | 2 |

68개 이미지의 Encoder Features는 적재 전후 모두 Byte 단위로 같았다. 모든 값은 유한했다.
이 비교는 해당 3개 입력의 Encoder 결과만 검사한다. 전체 모델의 Task Quality 검사가 아니다.

| Component | Measured bytes | Decimal GB |
|---|---:|---:|
| E Vision Parameters | 1353100288 | 1.353 |
| 추가 Full-model Parameters | 16584333312 | 16.584 |
| 추가 모델의 LLM Parameters | 15231233024 | 15.231 |
| 추가 모델의 중복 Vision Parameters | 1353100288 | 1.353 |
| 추가 Engine의 고유 KV Storage | 8587837440 | 8.588 |

추가 Engine은 LLM만 로드한 경량 Worker가 아니다. 원본 Full-model과 중복 Vision Tower를 포함한다.
KV 요청 상한은 8GiB였다. 실제 Storage 크기는 위 표와 같다.
적재 후 실제 Encoder 입력 처리까지 마쳤을 때 Device Free Memory는 43844042752 bytes였다.
Device Total Memory는 85028372480 bytes였다.

E의 Peak Allocated Memory는 6604047872 bytes에서 7117387776 bytes로 변했다.
이 값은 각 구간까지의 Torch Allocator 최대값이다. 순수 Activation 또는 Workspace의 독점 크기가 아니다.
메모리 차이를 Encoder의 계산시간 변화로 해석하지 않는다.

추가 Engine은 실제 요청의 Prefill을 처리하지 않았다. 추가 Quantum 실행도 없었다.
따라서 `ENCODE_SAFE_HEADROOM_NOT_MEASURED`다. 최대 Encoder Batch Capacity도 미측정이다.
Online ITL, Encode p95 비열등성, Queue 안정성, Serving Speedup을 이 결과로 판정하지 않는다.

원시 결과는 `analysis/encoder_LLM_MEMORY_C16.json`에 있다.
요약은 `analysis/encoder_LLM_memory_C16_summary_v0.json`에 있다.
Figure와 대응 CSV는 `figures/encoder_LLM_memory_C16_v0.{png,pdf,csv}`다.
모든 경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.

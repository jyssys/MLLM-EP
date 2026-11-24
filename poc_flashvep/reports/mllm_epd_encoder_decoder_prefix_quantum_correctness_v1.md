# E/D 실제 Prefix Quantum 정합성 — C22R1

E Helper와 D Helper의 Prefix 검사 10/10이 통과했다. 증거 수준은 `CORRECTNESS_ONLY`다.
실행 비용, 추가 Prefill의 Online 안전성 또는 Serving 개선은 아직 입증하지 않았다.

## 원본 계산과 비교 범위

Qwen2.5-VL-7B-Instruct 원본 BF16과 Processor를 사용했다.
모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
원본 Calibration Request6은 Input2961 Tokens와 이미지 2개다. 원문과 이미지 순서는 유지했다.
E는 두 이미지 모두 실제로 Encoding했다. Helper는 명시한 Prefix Query만 계산했다.
GPU4는 E와 추가 원본 Full-model Helper를 함께 실행했다. GPU6은 같은 Helper만 실행했다.
Helper KV는 각각 8GiB다. B8192/S1024, 실제 요청 Batch1을 사용했다.
각 Prefix는 같은 GPU와 같은 Query Chunk 경계의 Full Teacher와 비교했다.
원래 Image Metadata, Prepared Embeddings, MRoPE와 Position IDs를 검사했다.
28개 Layer의 해당 Prefix KV가 모두 Byte 단위로 일치했다.
각 요청의 1 Token 임시 Sample은 폐기했다. 완성된 Task Answer나 Accuracy 측정이 아니다.
CPU Tensor 복사가 있으므로 이 실행의 Timing은 성능 결과로 사용하지 않는다.

| Helper | GPU | Prefix Tokens | Strict KV Layers | Prepared Input / MRoPE |
|---|---:|---:|---:|---|
| E | 4 | 128 | 28/28 | PASS |
| E | 4 | 256 | 28/28 | PASS |
| E | 4 | 512 | 28/28 | PASS |
| E | 4 | 1024 | 28/28 | PASS |
| E | 4 | 2048 | 28/28 | PASS |
| D | 6 | 128 | 28/28 | PASS |
| D | 6 | 256 | 28/28 | PASS |
| D | 6 | 512 | 28/28 | PASS |
| D | 6 | 1024 | 28/28 | PASS |
| D | 6 | 2048 | 28/28 | PASS |

## 버퍼 수명 실패와 수정

C22의 첫 Probe는 ACK 대기 때문에 종료했다. 이 실패를 보존했다.
원래 Inline 정책은 Prefix에서 사용하지 않는 두 번째 이미지에도 Descriptor를 발급했다.
사용하지 않은 Descriptor에는 Clone ACK가 없었다. 안전 검사가 버퍼 해제를 거부했다.
C22R1은 이 고립 Probe에만 기존 On-demand Descriptor 경로를 적용했다.
실제로 필요한 이미지에만 Descriptor를 발급했다. 발급한 모든 Descriptor의 Clone 완료 ACK는 계속 필수다.
원본 E의 전체 Encoding은 생략하지 않았다. 버퍼를 조기에 해제하지 않았다.
실제 Registry 코드의 CPU 재현은 수정 전 실패했고 수정 후 통과했다.
GPU 재검증 10개도 통과했다. 요청 종료 후 Live Images와 Outstanding Exports가 모두 0인지 검사했다.
이 변경은 Native의 Early Inline 성능 정책 변경이 아니다. 공통 성능 비교로 해석하지 않는다.

## 한계와 다음 측정

이 결과는 KV가 없는 Prefix Prefill이다. Prefix KV를 가져오는 Suffix Continuation과 다르다.
E/D Safe Quantum, Encode/Decode 비열등성, 최대 Encoder Batch는 미측정이다.
동일 Request를 E/P/D에서 병렬 처리한 결과가 아니다. Extra Prefill은 아직 Online 요청에 삽입하지 않았다.
Task Quality와 Stock Compiled Token C3 Gate도 별도로 남아 있다.
원시 비교는 `analysis/prefix_QUANTA_C22R1_*_comparison.json`에 있다.
실패는 `state/prefix_QUANTA_C22_failure.json`과 대응 로그에 보존했다.
대응 CSV는 `analysis/prefix_QUANTA_C22R1_cases_v0.csv`다.
모든 경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.

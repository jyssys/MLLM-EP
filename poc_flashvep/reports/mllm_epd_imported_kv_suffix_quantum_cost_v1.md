# 실제 Prefix KV를 가져온 Suffix Quantum 비용

전체 증거 수준은 `DIAGNOSTIC`다. 독립 초기화는 1회다.
15개 요청·Quantum 조합의 Strict 상태 검사를 먼저 통과했다. 이후 Cost 표본을 별도로 측정했다.
각 Cost 요청은 실제 원래 E, P Prefix, NIXL KV Import와 Full28 D Suffix를 실행했다.
Output1 Sample은 폐기했다. Natural EOS나 C3 Online Serving 결과가 아니다.

## 실제 비용

| Input | Suffix | Clean Before HTTP p95 ms | Clean After HTTP p95 ms | Profile Forward p95 ms | Query Work Conserved |
|---:|---:|---:|---:|---:|---|
| 4831 | 128 | 79.595 | 125.908 | 60.174 | True |
| 4831 | 256 | 170.820 | 106.250 | 94.892 | True |
| 4831 | 512 | 125.540 | 98.524 | 31.202 | True |
| 4831 | 1024 | 90.701 | 97.451 | 49.615 | True |
| 4831 | 2048 | 142.405 | 107.979 | 63.611 | True |
| 2961 | 128 | 68.390 | 74.846 | 28.934 | True |
| 2961 | 256 | 180.834 | 74.811 | 150.198 | True |
| 2961 | 512 | 63.221 | 105.576 | 50.093 | True |
| 2961 | 1024 | 77.937 | 175.695 | 140.152 | True |
| 2961 | 2048 | 89.388 | 95.190 | 76.520 | True |
| 27840 | 128 | 125.939 | 200.343 | 45.761 | True |
| 27840 | 256 | 144.730 | 148.792 | 63.309 | True |
| 27840 | 512 | 132.668 | 137.324 | 45.989 | True |
| 27840 | 1024 | 154.459 | 160.078 | 65.192 | True |
| 27840 | 2048 | 201.262 | 224.265 | 154.926 | True |

D HTTP 구간은 Nonstreaming 한 Token 응답 완료까지다. SSE의 첫 Token TTFT와 정의가 다르다.
KV Import·Embedding 소비·Admission·Forward·Sampling·Delivery가 이 구간에 들어 있다.
Forward Event는 Stream Enclosure다. GPU-exclusive Kernel이나 DMA 시간으로 바꾸지 않는다.
각 Cell의 Clean/Profile/Clean 표본은 단계별 9개다. 반복 9개를 독립 Server Start9회로 계산하지 않았다.
고정한 30개 p50/p95 검사 중 하나라도 실패하면 전체 계측을 Diagnostic으로 유지한다.
원래 전체 Observer Gate는 False다. 실제 Query Conservation은 True다.
요청이 없는 정리 Execute 944개는 따로 보존했다. 해당 비용을 Suffix에 임의로 배분하지 않았다.

## Physical Gap 비교의 한계

별도의 QPS0.5 Main100 D Kernel Gap과 비용을 비교한 값은 `OFFLINE_ORACLE`이다.
미래 Gap 길이를 안다고 가정했다. KV 준비·Stream·메모리·새 Arrival·Deadline을 검사하지 않았다.
비용에는 Prefix KV Import가 포함된다. 실제 미리 준비된 KV에 대한 더 작은 Service 비용이라고 바꾸지 않는다.
이 물리적 Fit Rate로 Decode-safe Window, Active Decode 삽입 또는 실제 Speedup을 주장하지 않는다.
E의 Prefix Probe와 이 D Suffix는 서로 다른 작업이다. E/D가 같은 Suffix를 처리한 성능 비교가 아니다.

## 재현 계약

Original BF16 Qwen2.5-VL-7B, BI0, Full28과 원본 Vision/Processor를 유지했다.
Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다. GPU4/5/6은 E/P/D다.
B8192/S1024, KV42GiB, Block128과 공통 Compiled Reduction 설정이다.
Calibration Request0/6/31의 원문을 유지했다. Input 길이를 맞추기 위한 Padding이나 Truncation은 하지 않았다.
논리적 Prefix KV Bytes는 Structural Accounting이다. 실제 전송된 Padding·DMA Bytes는 미측정이다.
원시 요청과 검사 CSV는 `analysis/suffix_QUANTA_C41R3_*_v0`에 있다.
실제 Forward Event는 `raw/gpu_enclosures/suffix_QUANTA_C41R3/decode/`에 있다.
Figure와 대응 CSV는 `figures/suffix_QUANTA_C41R3_cost_v0`에 있다.

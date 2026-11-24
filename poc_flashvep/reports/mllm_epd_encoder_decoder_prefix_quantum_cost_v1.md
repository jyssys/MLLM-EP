# E/D Prefix Quantum 비용 — 진단용 측정

두 Arm의 전체 Clean/Profile/Clean Gate는 실패했다. 비용은 모두 `DIAGNOSTIC`으로 보존했다.
E는 주로 비계측 실행 간 변동 때문에 실패했다. D도 일부 Quantum의 Tail/변동 기준을 통과하지 못했다.
이 결과로 Safe Quantum, Online Speedup 또는 Encoder/Decode 비열등성을 주장하지 않는다.

## 계산과 측정 계약

Original Qwen2.5-VL-7B BF16, BI0, Full28과 원본 Processor를 사용했다.
모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
동일 Calibration Request6의 Input2961 Tokens와 이미지 2개를 유지했다.
E는 매 요청에서 원본 이미지 2개를 실제 Encoding했다. Helper는 명시한 Prefix만 처리했다.
GPU4는 E와 추가 원본 Full-model Helper를 함께 실행했다. GPU6은 같은 Helper를 독립적으로 실행했다.

B8192/S1024, Helper KV8GiB와 실제 Batch1을 사용했다. Prefix에는 가져온 KV가 없다.
Output1 Token은 폐기했다. 완성된 Task Answer와 Natural EOS Serving이 아니다.
조건별 독립 초기화는 1회다. 같은 초기화 안의 반복 9개를 독립 Start9회로 계산하지 않았다.
E/D 초기화도 서로 다르다. E/D Paired CI 또는 물리적 GPU 우위로 해석하지 않는다.

## 전체 Cell 결과

HTTP Median은 비계측 표본 18개다. Execute p50/p95는 계측 표본 9개다.
Execute Event에는 Host Gap, Dependency와 Embedding Import가 포함될 수 있다.
Import Host 시간은 Execute와 중첩될 수 있다. 두 값을 더해 Total GPU Time으로 만들지 않는다.

| Helper | Prefix | Clean HTTP p50 ms | Execute p50 ms | Execute p95 ms | Import Host p50 ms | Local HTTP Gates | Arm Gate |
|---|---:|---:|---:|---:|---:|---|---|
| E | 128 | 326.967 | 26.745 | 29.888 | 7.198 | FAIL | FAIL |
| E | 256 | 328.144 | 28.796 | 53.958 | 7.494 | FAIL | FAIL |
| E | 512 | 343.915 | 27.540 | 48.677 | 7.568 | FAIL | FAIL |
| E | 1024 | 349.651 | 51.347 | 58.500 | 7.691 | FAIL | FAIL |
| E | 2048 | 372.355 | 56.689 | 62.254 | 7.378 | FAIL | FAIL |
| D | 128 | 351.203 | 35.173 | 51.954 | 16.031 | PASS | FAIL |
| D | 256 | 352.246 | 34.067 | 62.552 | 15.669 | FAIL | FAIL |
| D | 512 | 349.550 | 35.431 | 45.739 | 15.730 | FAIL | FAIL |
| D | 1024 | 370.403 | 54.152 | 62.285 | 15.764 | PASS | FAIL |
| D | 2048 | 385.951 | 66.848 | 97.693 | 16.359 | FAIL | FAIL |

일부 D Cell의 두 HTTP 검사가 통과해도 전체 Arm을 PASS로 바꾸지 않았다.
실패 Cell과 Outlier는 모두 CSV에 남겼다. 유리한 표본만 제거하지 않았다.

## 실제 Embedding 전달 단위

모든 Quantum은 첫 이미지 Span15:2706 안에서 끝난다. 해당 이미지의 Visual Embedding은 2691×3584 BF16이다.
실제 Import Log는 모든 표본에서 이미지 1개와 19,289,088 Bytes를 기록했다.
E/D 각 140개 HTTP 구간과 같은 Monotonic Clock의 Import 구간을 일대일로 연결했다.
각 Arm의 Consumer PID가 하나인지 확인했다. Clone 완료 ACK도 모든 행에서 유지했다.
따라서 Prefix128을 처리해도 이 Runtime은 첫 이미지 Embedding 전체를 Import했다.

이 관측은 Runtime의 Image-level 전달 단위에 관한 것이다. Visual Token의 계산이 본질적으로 더 비싸다는 증거가 아니다.
이미지 내부 Exact Split의 수학적 정합성과 전달 효율은 다른 문제다.
전달 단위를 변경하는 Modality-aware Method는 구현하지 않았다. Novelty도 주장하지 않는다.

## 정리 단계와 실패 보존

E/D 각 계측에는 요청 실행 45개와 요청이 없는 Engine 정리 실행 90개가 있었다.
정리 단계의 Event는 별도 항목으로 보존했다. Prefill GPU 비용에 임의로 배분하지 않았다.
E의 최초 분석은 이 정리 단계를 예상하지 못해 종료했다. 원시 측정은 이미 완료된 상태였다.
실제 Event 패턴의 CPU Red/Green 검사 뒤, E 데이터는 GPU 재실행 없이 복구했다.
D 비용만 별도 실행했다. E의 실패 Gate와 기존 Raw를 덮어쓰지 않았다.

## 한계와 재현 파일

Suffix Continuation은 Prefix Length와 가져온 KV에 따라 비용이 달라진다. 이 Prefix 비용으로 대체하지 않는다.
실제 Encode/Decode Queue가 있는 Online Intervention은 `NOT_RUN`이다.
Future Idle Window에 맞춘 안전한 Quantum 또는 Encoder/Decode SLO 보호도 미검증이다.
요청·Gate 원시는 `analysis/prefix_QUANTA_C26R{1_E,2_D}_CPC_{requests,qualification}.csv`에 있다.
비용과 정리 Event는 대응 `_cost.json`에 있다.
결합 CSV는 `analysis/prefix_quanta_E_D_C26_{summary,embedding_imports}_v0.csv`다.
Figure는 `figures/prefix_quanta_E_D_C26_cost_v0.{png,pdf}`다. Source SHA와 GPU 정보는 Provenance JSON에 있다.
모든 경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.

# H200 독립 Stage Capacity

같은 Main100 원문과 원래 Encode·P·D Engine을 사용한다. E는 실제 원본 이미지의 동시 Encoding을 사용한다. P에는 앞서 원래 E가 준비한 Embedding을 제공한다. P 측정은 D Backpressure가 없는 로컬 KV 반환 경계를 사용한다. D는 원래 P의 NIXL KV와 동결 5,269 Output Token 분포를 사용하며 E/P 준비 구간을 분모에서 제외한다. D-only B1과 Unrestricted는 따로 측정한다.

| Stage | Run | 완료 요청 | 실제 최대 Batch | req/s | tok/s | 실제 E Rows/s | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 원래 E | c_capacity | 100 | 195 | 3.92 | 43,009.65 | 43,009.65 | ISOLATED_GPU_MEASUREMENT |
| Embedding-ready P | c_capacity | 100 | 41 | 3.18 | 35,036.55 | — | ISOLATED_GPU_MEASUREMENT |
| Embedding-ready P | c_capacity_prefill | 100 | 74 | 3.17 | 35,018.63 | — | ISOLATED_GPU_MEASUREMENT |
| NIXL D Batch1 | c_capacity_decode-b1 | 100 | 1 | 3.32 | 174.95 | — | ISOLATED_GPU_MEASUREMENT |
| NIXL D Unrestricted | c_capacity_decode-free | 100 | 12 | 7.19 | 379.06 | — | ISOLATED_GPU_MEASUREMENT |

E와 P의 최대 Batch는 각각 실제 Encoder 출력 이미지 수와 Prepared Prefill 요청 수다. D는 Actual Prepared Decode 요청 수다. E의 `tok/s`는 Manifest Visual Row 기준이며 실제 출력을 따로 기록한 경우 `실제 E Rows/s`에서 확인한다. P는 Input Token, D는 Output Token이다. Cohort 처리 시간은 각 Stage 첫 제출부터 마지막 완료까지이며 Cohort의 외부 준비 및 Drain 구간 정의를 Raw에 남긴다. Counter 오버헤드와 Steady State는 별개의 Gate다.

| Stage | 관측 | 값 |
| --- | --- | --- |
| 원래 E | 실제 E 이미지/s | 49.09 |
| Embedding-ready P | 격리 P Kernel Busy (진단) | NOT_MEASURED |
| Embedding-ready P | 격리 P Kernel Busy (진단) | 92.20% |
| NIXL D Batch1 | Client ITL p99 / TPOT p95 (ms) | 5.42 / 4.86 |
| NIXL D Batch1 | 실제 P KV Export Tokens p50 / p95 | 9,867.00 / 25,726.60 |
| NIXL D Unrestricted | Client ITL p99 / TPOT p95 (ms) | 23.20 / 10.27 |
| NIXL D Unrestricted | 실제 P KV Export Tokens p50 / p95 | 9,867.00 / 25,726.60 |

### Online / Isolated P 비교

| Run/Cohort | QPS | 단위 | Online P | Isolated P | Proximity | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `a_common/native_common_free_2` | 2.0 | requests_per_second | 1.68 | 3.18 | 52.73% | DIAGNOSTIC |
| `a_common/native_common_free_2` | 2.0 | tokens_per_second | 17,608.38 | 35,036.55 | 50.26% | DIAGNOSTIC |
| `a_common/native_common_free_2` | 2.0 | requests_per_second | 1.68 | 3.17 | 52.76% | DIAGNOSTIC |
| `a_common/native_common_free_2` | 2.0 | tokens_per_second | 17,608.38 | 35,018.63 | 50.28% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_0p5` | 0.5 | requests_per_second | 0.54 | 3.18 | 16.91% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_0p5` | 0.5 | tokens_per_second | 5,681.27 | 35,036.55 | 16.22% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_0p5` | 0.5 | requests_per_second | 0.54 | 3.17 | 16.92% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_0p5` | 0.5 | tokens_per_second | 5,681.27 | 35,018.63 | 16.22% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_1` | 1.0 | requests_per_second | 1.00 | 3.18 | 31.52% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_1` | 1.0 | tokens_per_second | 10,612.28 | 35,036.55 | 30.29% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_1` | 1.0 | requests_per_second | 1.00 | 3.17 | 31.54% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_1` | 1.0 | tokens_per_second | 10,612.28 | 35,018.63 | 30.30% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_2` | 2.0 | requests_per_second | 1.71 | 3.18 | 53.88% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_2` | 2.0 | tokens_per_second | 17,993.31 | 35,036.55 | 51.36% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_2` | 2.0 | requests_per_second | 1.71 | 3.17 | 53.90% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_2` | 2.0 | tokens_per_second | 17,993.31 | 35,018.63 | 51.38% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_3` | 3.0 | requests_per_second | 2.18 | 3.18 | 68.78% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_3` | 3.0 | tokens_per_second | 24,449.31 | 35,036.55 | 69.78% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_3` | 3.0 | requests_per_second | 2.18 | 3.17 | 68.81% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_3` | 3.0 | tokens_per_second | 24,449.31 | 35,018.63 | 69.82% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_4` | 4.0 | requests_per_second | 2.91 | 3.18 | 91.70% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_4` | 4.0 | tokens_per_second | 29,526.38 | 35,036.55 | 84.27% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_4` | 4.0 | requests_per_second | 2.91 | 3.17 | 91.75% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_4` | 4.0 | tokens_per_second | 29,526.38 | 35,018.63 | 84.32% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_5` | 5.0 | requests_per_second | 3.55 | 3.18 | 111.76% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_5` | 5.0 | tokens_per_second | 36,542.29 | 35,036.55 | 104.30% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_5` | 5.0 | requests_per_second | 3.55 | 3.17 | 111.82% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_5` | 5.0 | tokens_per_second | 36,542.29 | 35,018.63 | 104.35% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_6` | 6.0 | requests_per_second | 3.39 | 3.18 | 106.61% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_6` | 6.0 | tokens_per_second | 33,072.78 | 35,036.55 | 94.40% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_6` | 6.0 | requests_per_second | 3.39 | 3.17 | 106.66% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_6` | 6.0 | tokens_per_second | 33,072.78 | 35,018.63 | 94.44% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_8` | 8.0 | requests_per_second | 3.79 | 3.18 | 119.22% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_8` | 8.0 | tokens_per_second | 37,471.70 | 35,036.55 | 106.95% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_8` | 8.0 | requests_per_second | 3.79 | 3.17 | 119.28% | DIAGNOSTIC |
| `c_native_b1_screen/native_common_b1_8` | 8.0 | tokens_per_second | 37,471.70 | 35,018.63 | 107.01% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_1` | 1.0 | requests_per_second | 1.00 | 3.18 | 31.52% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_1` | 1.0 | tokens_per_second | 10,612.28 | 35,036.55 | 30.29% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_1` | 1.0 | requests_per_second | 1.00 | 3.17 | 31.54% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_1` | 1.0 | tokens_per_second | 10,612.28 | 35,018.63 | 30.30% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_3` | 3.0 | requests_per_second | 2.13 | 3.18 | 67.06% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_3` | 3.0 | tokens_per_second | 23,618.60 | 35,036.55 | 67.41% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_3` | 3.0 | requests_per_second | 2.13 | 3.17 | 67.09% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_3` | 3.0 | tokens_per_second | 23,618.60 | 35,018.63 | 67.45% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_5` | 5.0 | requests_per_second | 3.37 | 3.18 | 106.03% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_5` | 5.0 | tokens_per_second | 34,548.84 | 35,036.55 | 98.61% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_5` | 5.0 | requests_per_second | 3.37 | 3.17 | 106.09% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_5` | 5.0 | tokens_per_second | 34,548.84 | 35,018.63 | 98.66% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_8` | 8.0 | requests_per_second | 3.50 | 3.18 | 110.05% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_8` | 8.0 | tokens_per_second | 32,045.67 | 35,036.55 | 91.46% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_8` | 8.0 | requests_per_second | 3.50 | 3.17 | 110.10% | DIAGNOSTIC |
| `c_native_free_screen/native_common_free_8` | 8.0 | tokens_per_second | 32,045.67 | 35,018.63 | 91.51% | DIAGNOSTIC |
| `d_profile_b1_q3/native_common_b1_3_profile` | 3.0 | requests_per_second | 2.08 | 3.18 | 65.34% | DIAGNOSTIC |
| `d_profile_b1_q3/native_common_b1_3_profile` | 3.0 | tokens_per_second | 23,353.63 | 35,036.55 | 66.66% | DIAGNOSTIC |
| `d_profile_b1_q3/native_common_b1_3_profile` | 3.0 | requests_per_second | 2.08 | 3.17 | 65.37% | DIAGNOSTIC |
| `d_profile_b1_q3/native_common_b1_3_profile` | 3.0 | tokens_per_second | 23,353.63 | 35,018.63 | 66.69% | DIAGNOSTIC |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | 4.0 | requests_per_second | 3.06 | 3.18 | 96.29% | DIAGNOSTIC |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | 4.0 | tokens_per_second | 32,126.07 | 35,036.55 | 91.69% | DIAGNOSTIC |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | 4.0 | requests_per_second | 3.06 | 3.17 | 96.34% | DIAGNOSTIC |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | 4.0 | tokens_per_second | 32,126.07 | 35,018.63 | 91.74% | DIAGNOSTIC |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | 5.0 | requests_per_second | 3.55 | 3.18 | 111.76% | DIAGNOSTIC |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | 5.0 | tokens_per_second | 36,542.29 | 35,036.55 | 104.30% | DIAGNOSTIC |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | 5.0 | requests_per_second | 3.55 | 3.17 | 111.82% | DIAGNOSTIC |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | 5.0 | tokens_per_second | 36,542.29 | 35,018.63 | 104.35% | DIAGNOSTIC |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | 5.0 | requests_per_second | 3.64 | 3.18 | 114.63% | DIAGNOSTIC |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | 5.0 | tokens_per_second | 36,839.15 | 35,036.55 | 105.14% | DIAGNOSTIC |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | 5.0 | requests_per_second | 3.64 | 3.17 | 114.69% | DIAGNOSTIC |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | 5.0 | tokens_per_second | 36,839.15 | 35,018.63 | 105.20% | DIAGNOSTIC |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | 6.0 | requests_per_second | 3.50 | 3.18 | 110.05% | DIAGNOSTIC |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | 6.0 | tokens_per_second | 36,235.72 | 35,036.55 | 103.42% | DIAGNOSTIC |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | 6.0 | requests_per_second | 3.50 | 3.17 | 110.10% | DIAGNOSTIC |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | 6.0 | tokens_per_second | 36,235.72 | 35,018.63 | 103.48% | DIAGNOSTIC |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | 5.0 | requests_per_second | 3.64 | 3.18 | 114.63% | DIAGNOSTIC |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | 5.0 | tokens_per_second | 36,697.50 | 35,036.55 | 104.74% | DIAGNOSTIC |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | 5.0 | requests_per_second | 3.64 | 3.17 | 114.69% | DIAGNOSTIC |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | 5.0 | tokens_per_second | 36,697.50 | 35,018.63 | 104.79% | DIAGNOSTIC |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | 5.0 | requests_per_second | 3.55 | 3.18 | 111.76% | DIAGNOSTIC |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | 5.0 | tokens_per_second | 36,484.30 | 35,036.55 | 104.13% | DIAGNOSTIC |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | 5.0 | requests_per_second | 3.55 | 3.17 | 111.82% | DIAGNOSTIC |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | 5.0 | tokens_per_second | 36,484.30 | 35,018.63 | 104.19% | DIAGNOSTIC |

Stage 분리는 병목 진단이다. P의 로컬 KV 반환과 Online KV Export·NIXL 수명, Actual Batch와 Attention Geometry 차이는 같은 조건이 확인될 때까지 보존한다. `DIAGNOSTIC` Proximity를 포화 증명으로 단독 사용하지 않는다.


| Run | Cohort | Evidence | 원인 |
| --- | --- | --- | --- |
| c_capacity | 초기화 | FAILED_EXECUTION | D-only replay failed: RuntimeError: {"error":{"message":"Handoff belongs to another input contract","type":"InternalServerError","param":null,"code":500}} |

원시는 각 Run의 `analysis/capacity_*.json`, `raw/capacity_*` 및 `stage_capacity.csv`, `service_proximity.csv`에 있다.

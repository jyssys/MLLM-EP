# H200 Decode Idle과 Prefix-ready 정렬

`D_at_P_ready`는 P 응답 직후 D의 Active HTTP·Pending Coordinator 요청을 기록한다. Native 관측은 Full Prefill 완료 시점이다. 실제 Static/Idle Token Split에 적용된 행에서만 이 값이 Suffix 이전 Prefix 완료의 실제 관측이다. HTTP는 Worker의 실제 D Prepared Decode와 물리적 CUDA Kernel Idle을 보수적으로 근사한다.

| Run/Cohort | Method | QPS | 관측 건 | D HTTP 비활성 | D 비활성 및 대기 없음 | D 대기 p50 ms | p95 ms | p99 ms | 실제 Split 건 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a_common/native_common_free_2` | native-common-free | 2.0 | 100 | 56.00% | 56.00% | — | — | — | 0 |
| `a_strong_v2/native_strong_free_2` | native-strong-free | 2.0 | 100 | 57.00% | 57.00% | — | — | — | 0 |
| `b_common_free/native_common_free_2_batch_clean_after` | native-common-free | 2.0 | 100 | 56.00% | 56.00% | — | — | — | 0 |
| `b_common_free/native_common_free_2_batch_clean_before` | native-common-free | 2.0 | 100 | 56.00% | 56.00% | — | — | — | 0 |
| `b_common_free/native_common_free_2_observed` | native-common-free | 2.0 | 100 | 56.00% | 56.00% | — | — | — | 0 |
| `b_idle10/idle10_b1_2_batch_clean_after` | idle10-B1 | 2.0 | 100 | 39.00% | 39.00% | 98.92 | 1,361.36 | 1,663.12 | 36 |
| `b_idle10/idle10_b1_2_batch_clean_before` | idle10-B1 | 2.0 | 100 | 41.00% | 41.00% | 128.95 | 1,533.57 | 1,865.86 | 35 |
| `b_idle10/idle10_b1_2_observed` | idle10-B1 | 2.0 | 100 | 43.00% | 43.00% | 63.36 | 1,712.04 | 2,014.27 | 38 |
| `b_native_common/native_common_b1_2_batch_clean_before` | native-common-B1 | 2.0 | 100 | 50.00% | 50.00% | 0.47 | 956.81 | 1,536.41 | 0 |
| `b_native_common_v2/native_common_b1_2_batch_clean_after` | native-common-B1 | 2.0 | 100 | 43.00% | 43.00% | 51.30 | 1,118.89 | 1,656.62 | 0 |
| `b_native_common_v2/native_common_b1_2_batch_clean_before` | native-common-B1 | 2.0 | 100 | 43.00% | 43.00% | 66.56 | 1,289.76 | 1,659.80 | 0 |
| `b_native_common_v2/native_common_b1_2_observed` | native-common-B1 | 2.0 | 100 | 45.00% | 45.00% | 49.57 | 1,492.47 | 1,903.20 | 0 |
| `b_static10/static10_b1_2_batch_clean_after` | static10-B1 | 2.0 | 100 | 32.00% | 32.00% | 390.57 | 2,471.53 | 2,799.00 | 100 |
| `b_static10/static10_b1_2_batch_clean_before` | static10-B1 | 2.0 | 100 | 38.00% | 38.00% | 228.51 | 2,204.37 | 2,503.42 | 100 |
| `b_static10/static10_b1_2_observed` | static10-B1 | 2.0 | 100 | 37.00% | 37.00% | 165.84 | 2,428.25 | 2,719.41 | 100 |
| `b_strong_b1/native_strong_b1_2_batch_clean_after` | native-strong-B1 | 2.0 | 100 | 44.00% | 44.00% | 34.41 | 1,294.09 | 1,680.21 | 0 |
| `b_strong_b1/native_strong_b1_2_batch_clean_before` | native-strong-B1 | 2.0 | 100 | 46.00% | 46.00% | 45.90 | 1,140.27 | 1,663.77 | 0 |
| `b_strong_b1/native_strong_b1_2_observed` | native-strong-B1 | 2.0 | 100 | 46.00% | 46.00% | 32.74 | 1,060.85 | 1,600.81 | 0 |
| `c_native_b1_screen/native_common_b1_0p5` | native-common-B1 | 0.5 | 100 | 79.00% | 79.00% | 0.02 | 302.33 | 663.28 | 0 |
| `c_native_b1_screen/native_common_b1_1` | native-common-B1 | 1.0 | 100 | 67.00% | 67.00% | 0.02 | 562.66 | 1,014.06 | 0 |
| `c_native_b1_screen/native_common_b1_2` | native-common-B1 | 2.0 | 100 | 43.00% | 42.00% | 83.46 | 1,169.68 | 1,733.53 | 0 |
| `c_native_b1_screen/native_common_b1_3` | native-common-B1 | 3.0 | 100 | 28.00% | 28.00% | 319.40 | 1,873.31 | 2,957.26 | 0 |
| `c_native_b1_screen/native_common_b1_4` | native-common-B1 | 4.0 | 100 | 19.00% | 19.00% | 793.19 | 3,364.75 | 3,981.76 | 0 |
| `c_native_b1_screen/native_common_b1_5` | native-common-B1 | 5.0 | 100 | 11.00% | 11.00% | 1,632.24 | 4,045.42 | 4,222.21 | 0 |
| `c_native_b1_screen/native_common_b1_6` | native-common-B1 | 6.0 | 100 | 8.00% | 8.00% | 1,918.38 | 4,131.00 | 4,359.45 | 0 |
| `c_native_b1_screen/native_common_b1_8` | native-common-B1 | 8.0 | 100 | 12.00% | 12.00% | 1,105.50 | 3,723.73 | 3,811.96 | 0 |
| `c_native_free_screen/native_common_free_1` | native-common-free | 1.0 | 100 | 68.00% | 68.00% | — | — | — | 0 |
| `c_native_free_screen/native_common_free_3` | native-common-free | 3.0 | 100 | 47.00% | 47.00% | — | — | — | 0 |
| `c_native_free_screen/native_common_free_5` | native-common-free | 5.0 | 100 | 43.00% | 43.00% | — | — | — | 0 |
| `c_native_free_screen/native_common_free_8` | native-common-free | 8.0 | 100 | 41.00% | 41.00% | — | — | — | 0 |
| `d_profile_b1_q3/native_common_b1_3_clean_after` | native-common-B1 | 3.0 | 100 | 26.00% | 26.00% | 340.42 | 2,014.39 | 2,679.11 | 0 |
| `d_profile_b1_q3/native_common_b1_3_clean_before` | native-common-B1 | 3.0 | 100 | 28.00% | 28.00% | 281.08 | 1,825.61 | 2,755.04 | 0 |
| `d_profile_b1_q3/native_common_b1_3_profile` | native-common-B1 | 3.0 | 100 | 25.00% | 25.00% | 320.23 | 2,034.57 | 2,878.15 | 0 |
| `d_profile_b1_q4_sampled/native_common_b1_4_clean_after` | native-common-B1 | 4.0 | 100 | 15.00% | 15.00% | 644.67 | 2,799.92 | 3,397.87 | 0 |
| `d_profile_b1_q4_sampled/native_common_b1_4_clean_before` | native-common-B1 | 4.0 | 100 | 19.00% | 19.00% | 491.42 | 2,645.15 | 3,216.52 | 0 |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | native-common-B1 | 4.0 | 100 | 14.00% | 14.00% | 630.44 | 3,481.34 | 4,052.74 | 0 |
| `d_profile_b1_q5_deferred/native_common_b1_5_clean_after` | native-common-B1 | 5.0 | 100 | 9.00% | 9.00% | 1,811.71 | 4,061.70 | 4,254.93 | 0 |
| `d_profile_b1_q5_deferred/native_common_b1_5_clean_before` | native-common-B1 | 5.0 | 100 | 15.00% | 15.00% | 1,707.75 | 4,043.18 | 4,234.16 | 0 |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | native-common-B1 | 5.0 | 100 | 7.00% | 7.00% | 2,277.24 | 4,211.31 | 4,360.67 | 0 |
| `d_profile_b1_q5_sampled/native_common_b1_5_clean_after` | native-common-B1 | 5.0 | 100 | 11.00% | 11.00% | 1,146.70 | 3,412.74 | 3,507.48 | 0 |
| `d_profile_b1_q5_sampled/native_common_b1_5_clean_before` | native-common-B1 | 5.0 | 100 | 14.00% | 14.00% | 1,725.38 | 3,699.72 | 3,755.15 | 0 |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | native-common-B1 | 5.0 | 100 | 9.00% | 9.00% | 1,573.01 | 3,793.25 | 4,003.37 | 0 |
| `d_profile_b1_q6_sampled/native_common_b1_6_clean_after` | native-common-B1 | 6.0 | 100 | 8.00% | 8.00% | 1,938.07 | 4,215.56 | 4,443.65 | 0 |
| `d_profile_b1_q6_sampled/native_common_b1_6_clean_before` | native-common-B1 | 6.0 | 100 | 8.00% | 8.00% | 1,398.56 | 3,557.98 | 3,654.93 | 0 |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | native-common-B1 | 6.0 | 100 | 8.00% | 7.00% | 2,179.93 | 4,200.81 | 4,347.81 | 0 |
| `d_profile_free_q5_deferred/native_common_free_5_clean_after` | native-common-free | 5.0 | 100 | 40.00% | 40.00% | — | — | — | 0 |
| `d_profile_free_q5_deferred/native_common_free_5_clean_before` | native-common-free | 5.0 | 100 | 38.00% | 38.00% | — | — | — | 0 |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | native-common-free | 5.0 | 100 | 39.00% | 39.00% | — | — | — | 0 |
| `d_profile_free_q5_sampled/native_common_free_5_clean_after` | native-common-free | 5.0 | 100 | 42.00% | 42.00% | — | — | — | 0 |
| `d_profile_free_q5_sampled/native_common_free_5_clean_before` | native-common-free | 5.0 | 100 | 45.00% | 45.00% | — | — | — | 0 |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | native-common-free | 5.0 | 100 | 37.00% | 37.00% | — | — | — | 0 |

## Idle Window의 길이

| Run | Cohort | Idle 정의 | Window 수 | p50 ms | p95 ms | 160ms 이상 Duration | 225ms 이상 Duration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a_common | native_common_free_2 | No active D HTTP | 31 | 333.20 | 1,578.85 | 95.81% | 93.78% |
| a_common | native_common_free_2 | No active D HTTP or P-ready wait | 31 | 333.09 | 1,578.58 | 95.81% | 93.79% |
| a_strong_v2 | native_strong_free_2 | No active D HTTP | 32 | 351.63 | 1,351.74 | 97.30% | 93.95% |
| a_strong_v2 | native_strong_free_2 | No active D HTTP or P-ready wait | 32 | 351.40 | 1,351.67 | 97.31% | 93.96% |
| b_common_free | native_common_free_2_batch_clean_after | No active D HTTP | 31 | 346.09 | 1,301.40 | 96.56% | 91.99% |
| b_common_free | native_common_free_2_batch_clean_after | No active D HTTP or P-ready wait | 31 | 345.97 | 1,301.28 | 96.57% | 92.00% |
| b_common_free | native_common_free_2_batch_clean_before | No active D HTTP | 31 | 397.97 | 1,320.26 | 97.16% | 93.61% |
| b_common_free | native_common_free_2_batch_clean_before | No active D HTTP or P-ready wait | 31 | 397.86 | 1,320.19 | 97.16% | 93.61% |
| b_common_free | native_common_free_2_observed | No active D HTTP | 30 | 415.94 | 1,349.64 | 96.85% | 92.37% |
| b_common_free | native_common_free_2_observed | No active D HTTP or P-ready wait | 30 | 415.72 | 1,349.51 | 96.85% | 92.38% |
| b_idle10 | idle10_b1_2_batch_clean_after | No active D HTTP | 48 | 2.65 | 1,172.46 | 96.92% | 95.33% |
| b_idle10 | idle10_b1_2_batch_clean_after | No active D HTTP or P-ready wait | 23 | 489.72 | 1,258.45 | 97.06% | 95.47% |
| b_idle10 | idle10_b1_2_batch_clean_before | No active D HTTP | 49 | 0.71 | 1,100.20 | 98.65% | 95.68% |
| b_idle10 | idle10_b1_2_batch_clean_before | No active D HTTP or P-ready wait | 23 | 393.31 | 1,211.36 | 98.76% | 95.79% |
| b_idle10 | idle10_b1_2_observed | No active D HTTP | 46 | 24.60 | 1,152.24 | 97.46% | 96.03% |
| b_idle10 | idle10_b1_2_observed | No active D HTTP or P-ready wait | 25 | 411.76 | 1,242.49 | 97.56% | 96.12% |
| b_native_common | native_common_b1_2_batch_clean_before | No active D HTTP | 48 | 47.02 | 1,179.69 | 95.43% | 94.27% |
| b_native_common | native_common_b1_2_batch_clean_before | No active D HTTP or P-ready wait | 27 | 430.42 | 1,328.62 | 95.52% | 94.36% |
| b_native_common_v2 | native_common_b1_2_batch_clean_after | No active D HTTP | 48 | 21.18 | 1,285.27 | 97.32% | 94.81% |
| b_native_common_v2 | native_common_b1_2_batch_clean_after | No active D HTTP or P-ready wait | 24 | 481.42 | 1,728.56 | 97.42% | 94.91% |
| b_native_common_v2 | native_common_b1_2_batch_clean_before | No active D HTTP | 47 | 30.50 | 1,188.99 | 99.34% | 94.39% |
| b_native_common_v2 | native_common_b1_2_batch_clean_before | No active D HTTP or P-ready wait | 25 | 465.71 | 1,336.80 | 99.43% | 94.49% |
| b_native_common_v2 | native_common_b1_2_observed | No active D HTTP | 48 | 30.75 | 1,216.69 | 96.58% | 94.21% |
| b_native_common_v2 | native_common_b1_2_observed | No active D HTTP or P-ready wait | 25 | 438.25 | 1,250.11 | 96.68% | 94.31% |
| b_static10 | static10_b1_2_batch_clean_after | No active D HTTP | 44 | 9.20 | 1,161.46 | 98.05% | 91.75% |
| b_static10 | static10_b1_2_batch_clean_after | No active D HTTP or P-ready wait | 21 | 458.36 | 1,488.55 | 98.30% | 91.99% |
| b_static10 | static10_b1_2_batch_clean_before | No active D HTTP | 48 | 21.53 | 1,120.49 | 92.68% | 91.02% |
| b_static10 | static10_b1_2_batch_clean_before | No active D HTTP or P-ready wait | 24 | 315.99 | 1,218.46 | 92.81% | 91.15% |
| b_static10 | static10_b1_2_observed | No active D HTTP | 48 | 0.69 | 1,162.17 | 96.65% | 93.16% |
| b_static10 | static10_b1_2_observed | No active D HTTP or P-ready wait | 23 | 460.20 | 1,251.94 | 96.79% | 93.29% |
| b_strong_b1 | native_strong_b1_2_batch_clean_after | No active D HTTP | 49 | 52.77 | 1,190.62 | 97.14% | 93.20% |
| b_strong_b1 | native_strong_b1_2_batch_clean_after | No active D HTTP or P-ready wait | 27 | 433.70 | 1,335.88 | 97.23% | 93.29% |
| b_strong_b1 | native_strong_b1_2_batch_clean_before | No active D HTTP | 48 | 42.34 | 1,220.67 | 97.30% | 93.36% |
| b_strong_b1 | native_strong_b1_2_batch_clean_before | No active D HTTP or P-ready wait | 25 | 479.80 | 1,281.21 | 97.39% | 93.44% |
| b_strong_b1 | native_strong_b1_2_observed | No active D HTTP | 47 | 52.74 | 1,184.85 | 96.83% | 95.70% |
| b_strong_b1 | native_strong_b1_2_observed | No active D HTTP or P-ready wait | 27 | 417.52 | 1,234.12 | 96.92% | 95.80% |
| c_native_b1_screen | native_common_b1_0p5 | No active D HTTP | 60 | 1,140.81 | 4,421.47 | 99.69% | 99.51% |
| c_native_b1_screen | native_common_b1_0p5 | No active D HTTP or P-ready wait | 49 | 1,746.90 | 4,848.78 | 99.70% | 99.52% |
| c_native_b1_screen | native_common_b1_1 | No active D HTTP | 56 | 562.89 | 2,448.28 | 98.88% | 98.88% |
| c_native_b1_screen | native_common_b1_1 | No active D HTTP or P-ready wait | 41 | 810.84 | 2,488.65 | 98.91% | 98.91% |
| c_native_b1_screen | native_common_b1_2 | No active D HTTP | 47 | 57.40 | 1,235.46 | 97.56% | 93.79% |
| c_native_b1_screen | native_common_b1_2 | No active D HTTP or P-ready wait | 26 | 419.69 | 1,344.05 | 97.67% | 93.90% |
| c_native_b1_screen | native_common_b1_3 | No active D HTTP | 41 | 0.68 | 533.80 | 97.52% | 97.52% |
| c_native_b1_screen | native_common_b1_3 | No active D HTTP or P-ready wait | 13 | 406.18 | 871.76 | 97.83% | 97.83% |
| c_native_b1_screen | native_common_b1_4 | No active D HTTP | 41 | 0.60 | 1.18 | 0.00% | 0.00% |
| c_native_b1_screen | native_common_b1_5 | No active D HTTP | 27 | 0.63 | 111.64 | 82.07% | 82.07% |
| c_native_b1_screen | native_common_b1_5 | No active D HTTP or P-ready wait | 3 | 153.30 | 759.64 | 83.19% | 83.19% |
| c_native_b1_screen | native_common_b1_6 | No active D HTTP | 20 | 0.67 | 329.06 | 99.23% | 85.73% |
| c_native_b1_screen | native_common_b1_6 | No active D HTTP or P-ready wait | 3 | 300.12 | 817.94 | 100.00% | 86.40% |
| c_native_b1_screen | native_common_b1_8 | No active D HTTP | 16 | 0.65 | 42.39 | 94.79% | 0.00% |
| c_native_b1_screen | native_common_b1_8 | No active D HTTP or P-ready wait | 1 | 167.21 | 167.21 | 100.00% | 0.00% |
| c_native_free_screen | native_common_free_1 | No active D HTTP | 40 | 870.64 | 2,479.42 | 98.88% | 98.88% |
| c_native_free_screen | native_common_free_1 | No active D HTTP or P-ready wait | 40 | 870.40 | 2,479.32 | 98.88% | 98.88% |
| c_native_free_screen | native_common_free_3 | No active D HTTP | 19 | 255.53 | 1,459.46 | 94.37% | 92.25% |
| c_native_free_screen | native_common_free_3 | No active D HTTP or P-ready wait | 19 | 255.42 | 1,459.24 | 94.38% | 92.26% |
| c_native_free_screen | native_common_free_5 | No active D HTTP | 9 | 288.67 | 792.31 | 98.17% | 98.17% |
| c_native_free_screen | native_common_free_5 | No active D HTTP or P-ready wait | 9 | 288.33 | 792.16 | 98.17% | 98.17% |
| c_native_free_screen | native_common_free_8 | No active D HTTP | 5 | 478.00 | 888.90 | 94.53% | 94.53% |
| c_native_free_screen | native_common_free_8 | No active D HTTP or P-ready wait | 5 | 477.85 | 888.65 | 94.54% | 94.54% |
| d_profile_b1_q3 | native_common_b1_3_clean_after | No active D HTTP | 41 | 0.65 | 564.08 | 96.70% | 93.03% |
| d_profile_b1_q3 | native_common_b1_3_clean_after | No active D HTTP or P-ready wait | 12 | 472.50 | 1,069.84 | 97.01% | 93.32% |
| d_profile_b1_q3 | native_common_b1_3_clean_before | No active D HTTP | 42 | 0.66 | 569.34 | 96.93% | 96.93% |
| d_profile_b1_q3 | native_common_b1_3_clean_before | No active D HTTP or P-ready wait | 13 | 378.54 | 1,106.22 | 97.22% | 97.22% |
| d_profile_b1_q3 | native_common_b1_3_profile | No active D HTTP | 41 | 0.63 | 552.75 | 98.35% | 94.50% |
| d_profile_b1_q3 | native_common_b1_3_profile | No active D HTTP or P-ready wait | 11 | 463.12 | 1,023.09 | 98.78% | 94.92% |
| d_profile_b1_q4_sampled | native_common_b1_4_clean_after | No active D HTTP | 40 | 0.56 | 65.94 | 85.83% | 56.54% |
| d_profile_b1_q4_sampled | native_common_b1_4_clean_after | No active D HTTP or P-ready wait | 3 | 162.59 | 298.72 | 88.75% | 58.46% |
| d_profile_b1_q4_sampled | native_common_b1_4_clean_before | No active D HTTP | 40 | 0.62 | 35.74 | 93.10% | 93.10% |
| d_profile_b1_q4_sampled | native_common_b1_4_clean_before | No active D HTTP or P-ready wait | 3 | 298.83 | 305.72 | 96.58% | 96.58% |
| d_profile_b1_q4_sampled | native_common_b1_4_profile | No active D HTTP | 42 | 0.64 | 0.77 | 0.00% | 0.00% |
| d_profile_b1_q4_sampled | native_common_b1_4_profile | No active D HTTP or P-ready wait | 1 | 63.50 | 63.50 | 0.00% | 0.00% |
| d_profile_b1_q5_deferred | native_common_b1_5_clean_after | No active D HTTP | 27 | 0.64 | 331.85 | 98.58% | 98.58% |
| d_profile_b1_q5_deferred | native_common_b1_5_clean_after | No active D HTTP or P-ready wait | 2 | 563.67 | 644.85 | 100.00% | 100.00% |
| d_profile_b1_q5_deferred | native_common_b1_5_clean_before | No active D HTTP | 26 | 0.63 | 224.40 | 97.91% | 97.91% |
| d_profile_b1_q5_deferred | native_common_b1_5_clean_before | No active D HTTP or P-ready wait | 3 | 296.35 | 818.72 | 99.68% | 99.68% |
| d_profile_b1_q5_deferred | native_common_b1_5_profile | No active D HTTP | 26 | 0.69 | 46.03 | 90.81% | 90.81% |
| d_profile_b1_q5_deferred | native_common_b1_5_profile | No active D HTTP or P-ready wait | 2 | 428.53 | 760.86 | 93.08% | 93.08% |
| d_profile_b1_q5_sampled | native_common_b1_5_clean_after | No active D HTTP | 30 | 0.65 | 0.75 | 0.00% | 0.00% |
| d_profile_b1_q5_sampled | native_common_b1_5_clean_after | No active D HTTP or P-ready wait | 1 | 152.72 | 152.72 | 0.00% | 0.00% |
| d_profile_b1_q5_sampled | native_common_b1_5_clean_before | No active D HTTP | 27 | 0.65 | 203.80 | 97.84% | 97.84% |
| d_profile_b1_q5_sampled | native_common_b1_5_clean_before | No active D HTTP or P-ready wait | 2 | 355.84 | 414.56 | 100.00% | 100.00% |
| d_profile_b1_q5_sampled | native_common_b1_5_profile | No active D HTTP | 28 | 0.66 | 0.71 | 96.82% | 96.82% |
| d_profile_b1_q5_sampled | native_common_b1_5_profile | No active D HTTP or P-ready wait | 1 | 515.52 | 515.52 | 100.00% | 100.00% |
| d_profile_b1_q6_sampled | native_common_b1_6_clean_after | No active D HTTP | 20 | 0.68 | 529.21 | 94.41% | 94.41% |
| d_profile_b1_q6_sampled | native_common_b1_6_clean_after | No active D HTTP or P-ready wait | 4 | 289.60 | 782.80 | 95.13% | 95.13% |
| d_profile_b1_q6_sampled | native_common_b1_6_clean_before | No active D HTTP | 21 | 0.64 | 126.75 | 81.44% | 81.44% |
| d_profile_b1_q6_sampled | native_common_b1_6_clean_before | No active D HTTP or P-ready wait | 2 | 369.93 | 589.05 | 82.91% | 82.91% |
| d_profile_b1_q6_sampled | native_common_b1_6_profile | No active D HTTP | 21 | 0.68 | 142.96 | 81.99% | 81.99% |
| d_profile_b1_q6_sampled | native_common_b1_6_profile | No active D HTTP or P-ready wait | 2 | 425.84 | 680.59 | 83.23% | 83.23% |
| d_profile_free_q5_deferred | native_common_free_5_clean_after | No active D HTTP | 10 | 286.26 | 814.01 | 94.38% | 89.46% |
| d_profile_free_q5_deferred | native_common_free_5_clean_after | No active D HTTP or P-ready wait | 10 | 286.01 | 813.95 | 94.39% | 89.47% |
| d_profile_free_q5_deferred | native_common_free_5_clean_before | No active D HTTP | 12 | 174.00 | 448.40 | 89.71% | 80.40% |
| d_profile_free_q5_deferred | native_common_free_5_clean_before | No active D HTTP or P-ready wait | 12 | 173.70 | 448.12 | 89.75% | 80.44% |
| d_profile_free_q5_deferred | native_common_free_5_profile | No active D HTTP | 11 | 276.93 | 531.45 | 95.95% | 83.15% |
| d_profile_free_q5_deferred | native_common_free_5_profile | No active D HTTP or P-ready wait | 11 | 276.52 | 530.12 | 95.95% | 83.14% |
| d_profile_free_q5_sampled | native_common_free_5_clean_after | No active D HTTP | 12 | 174.91 | 693.48 | 89.87% | 78.07% |
| d_profile_free_q5_sampled | native_common_free_5_clean_after | No active D HTTP or P-ready wait | 12 | 174.60 | 693.33 | 89.87% | 78.09% |
| d_profile_free_q5_sampled | native_common_free_5_clean_before | No active D HTTP | 12 | 199.20 | 676.59 | 89.73% | 84.97% |
| d_profile_free_q5_sampled | native_common_free_5_clean_before | No active D HTTP or P-ready wait | 12 | 198.94 | 676.31 | 89.75% | 84.99% |
| d_profile_free_q5_sampled | native_common_free_5_profile | No active D HTTP | 10 | 285.72 | 563.72 | 95.79% | 76.16% |
| d_profile_free_q5_sampled | native_common_free_5_profile | No active D HTTP or P-ready wait | 10 | 285.34 | 563.62 | 95.80% | 76.18% |

| Run/Cohort | 교집합 | 중앙 ROI 대비 | P Busy 대비 | Evidence |
| --- | --- | --- | --- | --- |
| `d_profile_b1_q3/native_common_b1_3_profile` | P_kernel_D_kernel_idle | 28.64% | 46.03% | DIAGNOSTIC |
| `d_profile_b1_q3/native_common_b1_3_profile` | P_kernel_D_kernel_idle_no_D_HTTP_or_postprefill_wait | 16.18% | 26.01% | DIAGNOSTIC |
| `d_profile_b1_q3/native_common_b1_3_profile` | P_ready_before_HTTP_and_D_kernel_idle | 0.02% | — | DIAGNOSTIC |
| `d_profile_b1_q3/native_common_b1_3_profile` | P_kernel_and_P_ready_before_HTTP_and_D_kernel_idle | 0.01% | 0.02% | DIAGNOSTIC |
| `d_profile_b1_q3/native_common_b1_3_profile` | P_recent_scheduler_waiting_D_kernel_idle | 7.44% | — | DIAGNOSTIC |
| `d_profile_b1_q3/native_common_b1_3_profile` | P_kernel_and_recent_scheduler_waiting_D_kernel_idle | 6.66% | 10.70% | DIAGNOSTIC |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | P_kernel_D_kernel_idle | 12.27% | 15.39% | DIAGNOSTIC |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | P_kernel_D_kernel_idle_no_D_HTTP_or_postprefill_wait | 0.40% | 0.50% | DIAGNOSTIC |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | P_ready_before_HTTP_and_D_kernel_idle | 0.00% | — | DIAGNOSTIC |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | P_kernel_and_P_ready_before_HTTP_and_D_kernel_idle | 0.00% | 0.01% | DIAGNOSTIC |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | P_recent_scheduler_waiting_D_kernel_idle | 10.09% | — | DIAGNOSTIC |
| `d_profile_b1_q4_sampled/native_common_b1_4_profile` | P_kernel_and_recent_scheduler_waiting_D_kernel_idle | 9.39% | 11.78% | DIAGNOSTIC |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | P_kernel_D_kernel_idle | 18.47% | 19.16% | DIAGNOSTIC |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | P_kernel_D_kernel_idle_no_D_HTTP_or_postprefill_wait | 7.52% | 7.80% | DIAGNOSTIC |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | P_ready_before_HTTP_and_D_kernel_idle | 0.01% | — | DIAGNOSTIC |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | P_kernel_and_P_ready_before_HTTP_and_D_kernel_idle | 0.01% | 0.01% | DIAGNOSTIC |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | P_recent_scheduler_waiting_D_kernel_idle | 18.97% | — | DIAGNOSTIC |
| `d_profile_b1_q5_deferred/native_common_b1_5_profile` | P_kernel_and_recent_scheduler_waiting_D_kernel_idle | 18.47% | 19.16% | DIAGNOSTIC |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | P_kernel_D_kernel_idle | 15.54% | 16.05% | DIAGNOSTIC |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | P_kernel_D_kernel_idle_no_D_HTTP_or_postprefill_wait | 4.54% | 4.69% | DIAGNOSTIC |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | P_ready_before_HTTP_and_D_kernel_idle | 0.01% | — | DIAGNOSTIC |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | P_kernel_and_P_ready_before_HTTP_and_D_kernel_idle | 0.01% | 0.01% | DIAGNOSTIC |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | P_recent_scheduler_waiting_D_kernel_idle | 15.10% | — | DIAGNOSTIC |
| `d_profile_b1_q5_sampled/native_common_b1_5_profile` | P_kernel_and_recent_scheduler_waiting_D_kernel_idle | 14.76% | 15.24% | DIAGNOSTIC |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | P_kernel_D_kernel_idle | 19.72% | 21.02% | DIAGNOSTIC |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | P_kernel_D_kernel_idle_no_D_HTTP_or_postprefill_wait | 2.66% | 2.83% | DIAGNOSTIC |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | P_ready_before_HTTP_and_D_kernel_idle | 0.01% | — | DIAGNOSTIC |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | P_kernel_and_P_ready_before_HTTP_and_D_kernel_idle | 0.01% | 0.01% | DIAGNOSTIC |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | P_recent_scheduler_waiting_D_kernel_idle | 19.69% | — | DIAGNOSTIC |
| `d_profile_b1_q6_sampled/native_common_b1_6_profile` | P_kernel_and_recent_scheduler_waiting_D_kernel_idle | 19.10% | 20.36% | DIAGNOSTIC |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | P_kernel_D_kernel_idle | 38.33% | 40.84% | DIAGNOSTIC |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | P_kernel_D_kernel_idle_no_D_HTTP_or_postprefill_wait | 9.31% | 9.92% | DIAGNOSTIC |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | P_ready_before_HTTP_and_D_kernel_idle | 0.03% | — | DIAGNOSTIC |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | P_kernel_and_P_ready_before_HTTP_and_D_kernel_idle | 0.02% | 0.03% | DIAGNOSTIC |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | P_recent_scheduler_waiting_D_kernel_idle | 37.23% | — | DIAGNOSTIC |
| `d_profile_free_q5_deferred/native_common_free_5_profile` | P_kernel_and_recent_scheduler_waiting_D_kernel_idle | 35.89% | 38.23% | DIAGNOSTIC |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | P_kernel_D_kernel_idle | 37.34% | 40.17% | DIAGNOSTIC |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | P_kernel_D_kernel_idle_no_D_HTTP_or_postprefill_wait | 12.48% | 13.43% | DIAGNOSTIC |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | P_ready_before_HTTP_and_D_kernel_idle | 0.02% | — | DIAGNOSTIC |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | P_kernel_and_P_ready_before_HTTP_and_D_kernel_idle | 0.02% | 0.02% | DIAGNOSTIC |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | P_recent_scheduler_waiting_D_kernel_idle | 37.46% | — | DIAGNOSTIC |
| `d_profile_free_q5_sampled/native_common_free_5_profile` | P_kernel_and_recent_scheduler_waiting_D_kernel_idle | 36.26% | 39.01% | DIAGNOSTIC |

160ms와 225ms는 기존 H100 진단 비용에 근거해 사전 선택한 검토 길이다. Duration 비율 분모는 시리즈별 Idle Duration이다. P Busy와의 실제 교집합은 적격 CUPTI Trace와 공통 ROI에서만 확인한다. Future Idle Window를 알 수 있다는 가정을 둔 Fit은 `OFFLINE_ORACLE`이다.

- `discovery_gate_free_q5.json`은 Physical/Coordinator 교집합을 계산했다. 160ms 이상의 시간에서 P Busy Coverage는 27.22%다.
- `discovery_gate_free_q5_deferred.json`은 Physical/Coordinator 교집합을 계산했다. 160ms 이상의 시간에서 P Busy Coverage는 26.22%다.
- `discovery_gate_q3.json`은 Physical/Coordinator 교집합을 계산했다. 160ms 이상의 시간에서 P Busy Coverage는 6.99%다.
- `discovery_gate_q4.json`은 Physical/Coordinator 교집합을 계산했다. 160ms 이상의 시간에서 P Busy Coverage는 0.00%다.
- `discovery_gate_q5.json`은 Physical/Coordinator 교집합을 계산했다. 160ms 이상의 시간에서 P Busy Coverage는 4.72%다.
- `discovery_gate_q5_deferred.json`은 Physical/Coordinator 교집합을 계산했다. 160ms 이상의 시간에서 P Busy Coverage는 7.32%다.
- `discovery_gate_q5_sustained_r3.json`은 Physical/Coordinator 교집합을 계산했다. 160ms 이상의 시간에서 P Busy Coverage는 0.88%다.
- `discovery_gate_q5_sustained_r3_buffer2.json`은 Physical/Coordinator 교집합을 계산했다. 160ms 이상의 시간에서 P Busy Coverage는 1.64%다.
- `discovery_gate_q6.json`은 Physical/Coordinator 교집합을 계산했다. 160ms 이상의 시간에서 P Busy Coverage는 7.99%다.

재현 자료는 `decode_idle_windows.csv`, `kernel_overlap.csv`, `requests.csv`와 각 이미지·KV Owner Receipt다.

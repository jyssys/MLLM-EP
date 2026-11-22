# Final C3: 같은 Request의 Host 경계 차이

단위는 ms다. 값은 왼쪽 Arm에서 Reference를 뺀 평균이다.
각 행은 같은 Request ID, Prompt, Source, Arrival과 Pair를 비교한다.
6개 구간의 평균 차이는 Mean TTFT 차이에 합산된다.
이를 TTFT p95 또는 GPU-exclusive Operator의 분해라고 부르지 않는다.

| Workload / QPS | Arm − Reference | Paired Starts | Arrival→P Prepare | P Prepare→Ready | Ready→D Submit | D Submit→Prepare | D Prepare→First Worker | Worker→Client | Mean TTFT 차이 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heldout200/q1.8 | BestLayer − BestToken | 3 | 688.752 | 46.108 | 0.483 | -46.325 | -9.340 | 1.222 | 680.901 |
| Heldout200/q1.8 | BestHybrid − BestToken | 3 | 338.224 | 24.056 | 0.080 | -50.519 | 30.552 | -0.019 | 342.373 |
| Heldout200/q1.8 | BestHybrid − BestLayer | 3 | -350.527 | -22.053 | -0.403 | -4.195 | 39.893 | -1.242 | -338.527 |
| Heldout200/q1.8 | BestLayer − NativeStrong | 3 | -3742.477 | -293.049 | 0.280 | 166.644 | 144.381 | 1.216 | -3723.006 |
| Heldout200/q1.8 | BestLayer − NativeCommon2 | 3 | -3708.291 | -291.910 | 0.429 | 132.693 | 145.686 | 0.530 | -3720.862 |
| Heldout200/q1.8 | BestToken − NativeStrong | 3 | -4431.229 | -339.157 | -0.203 | 212.968 | 153.721 | -0.007 | -4403.906 |
| Heldout200/q1.8 | BestToken − NativeCommon3 | 3 | -4418.427 | -338.546 | -0.998 | 182.033 | 157.448 | 0.049 | -4418.440 |
| Heldout200/q1.8 | BestHybrid − NativeStrong | 3 | -4093.004 | -315.101 | -0.123 | 162.449 | 184.273 | -0.026 | -4061.533 |
| Heldout200/q1.8 | BestHybrid − NativeCommon0 | 3 | -4069.373 | -313.757 | -0.082 | 161.994 | 183.024 | -0.022 | -4038.216 |
| Heldout200/q2.8 | BestLayer − BestToken | 3 | 3717.913 | -11.394 | 0.294 | -36.917 | -26.869 | 0.301 | 3643.327 |
| Heldout200/q2.8 | BestHybrid − BestToken | 3 | 2277.713 | 5.404 | 0.192 | 3.144 | 65.017 | 0.063 | 2351.533 |
| Heldout200/q2.8 | BestHybrid − BestLayer | 3 | -1440.201 | 16.798 | -0.102 | 40.061 | 91.886 | -0.237 | -1291.794 |
| Heldout200/q2.8 | TokenCap2Control − BestToken | 3 | -6.910 | -4.559 | 0.119 | 37.323 | -13.590 | -0.011 | 12.373 |
| Heldout200/q2.8 | BestLayer − NativeStrong | 3 | -10559.979 | -217.438 | 0.555 | 295.850 | 153.273 | 0.354 | -10327.384 |
| Heldout200/q2.8 | BestLayer − NativeCommon1 | 3 | -10572.528 | -218.143 | 0.369 | 242.236 | 151.770 | 0.061 | -10396.233 |
| Heldout200/q2.8 | BestToken − NativeStrong | 3 | -14277.893 | -206.044 | 0.262 | 332.768 | 180.142 | 0.054 | -13970.711 |
| Heldout200/q2.8 | BestToken − NativeCommon2 | 3 | -14329.959 | -207.634 | 0.073 | 285.234 | 181.435 | 0.009 | -14070.841 |
| Heldout200/q2.8 | BestHybrid − NativeStrong | 3 | -12000.180 | -200.640 | 0.453 | 335.912 | 245.159 | 0.117 | -11619.178 |
| Heldout200/q2.8 | BestHybrid − NativeCommon2 | 3 | -12052.246 | -202.229 | 0.265 | 288.378 | 246.452 | 0.072 | -11719.308 |
| Heldout200/q2.8 | TokenCap2Control − NativeStrong | 3 | -14284.802 | -210.602 | 0.381 | 370.090 | 166.553 | 0.043 | -13958.338 |
| Heldout200/q2.8 | TokenCap2Control − NativeCommon1 | 3 | -14297.351 | -211.307 | 0.195 | 316.476 | 165.050 | -0.250 | -14027.187 |
| Main400/q1.0 | BestLayer − BestToken | 1 | 45.776 | 54.203 | 0.216 | -27.273 | -0.872 | -0.587 | 71.463 |
| Main400/q1.0 | BestHybrid − BestToken | 1 | 29.376 | 35.476 | 0.152 | -19.416 | 23.758 | -0.391 | 68.955 |
| Main400/q1.0 | BestHybrid − BestLayer | 1 | -16.400 | -18.727 | -0.064 | 7.856 | 24.630 | 0.197 | -2.508 |
| Main400/q1.0 | BestLayer − NativeStrong | 1 | -77.545 | -140.653 | -1.156 | 111.943 | 109.307 | 0.034 | 1.930 |
| Main400/q1.0 | BestLayer − NativeCommon2 | 1 | -76.357 | -141.457 | 0.191 | 82.835 | 112.517 | 0.038 | -22.232 |
| Main400/q1.0 | BestToken − NativeStrong | 1 | -123.321 | -194.857 | -1.372 | 139.216 | 110.179 | 0.621 | -69.533 |
| Main400/q1.0 | BestToken − NativeCommon3 | 1 | -123.855 | -195.874 | 0.009 | 110.483 | 114.418 | 0.460 | -94.360 |
| Main400/q1.0 | BestHybrid − NativeStrong | 1 | -93.945 | -159.381 | -1.220 | 119.800 | 133.938 | 0.231 | -0.577 |
| Main400/q1.0 | BestHybrid − NativeCommon0 | 1 | -96.112 | -163.902 | 0.005 | 124.132 | 137.469 | 0.158 | 1.750 |
| Main400/q1.8 | BestLayer − BestToken | 3 | 234.582 | 26.727 | -0.689 | -52.622 | -4.622 | -0.362 | 203.014 |
| Main400/q1.8 | BestHybrid − BestToken | 3 | 170.159 | 18.443 | -0.827 | -62.140 | 30.406 | -0.359 | 155.682 |
| Main400/q1.8 | BestHybrid − BestLayer | 3 | -64.423 | -8.284 | -0.137 | -9.518 | 35.028 | 0.003 | -47.332 |
| Main400/q1.8 | BestLayer − NativeStrong | 3 | -494.866 | -249.794 | -0.196 | 203.406 | 135.250 | 0.006 | -406.196 |
| Main400/q1.8 | BestLayer − NativeCommon2 | 3 | -493.212 | -250.750 | 0.075 | 138.593 | 133.456 | 0.016 | -471.821 |
| Main400/q1.8 | BestToken − NativeStrong | 3 | -729.448 | -276.522 | 0.493 | 256.028 | 139.872 | 0.368 | -609.209 |
| Main400/q1.8 | BestToken − NativeCommon3 | 3 | -727.923 | -276.226 | 0.739 | 194.738 | 139.618 | 0.367 | -668.686 |
| Main400/q1.8 | BestHybrid − NativeStrong | 3 | -559.290 | -258.078 | -0.333 | 193.888 | 170.277 | 0.009 | -453.528 |
| Main400/q1.8 | BestHybrid − NativeCommon0 | 3 | -553.262 | -257.144 | -0.310 | 190.461 | 167.527 | 0.006 | -452.722 |
| Main400/q2.8 | BestLayer − BestToken | 3 | 3392.370 | 17.782 | -1.126 | 132.244 | -1.878 | -1.440 | 3537.953 |
| Main400/q2.8 | BestHybrid − BestToken | 3 | 1981.293 | 22.318 | -1.033 | 117.836 | 131.134 | -1.488 | 2250.060 |
| Main400/q2.8 | BestHybrid − BestLayer | 3 | -1411.077 | 4.535 | 0.093 | -14.408 | 133.012 | -0.048 | -1287.893 |
| Main400/q2.8 | TokenCap2Control − BestToken | 3 | 17.415 | -5.162 | -1.018 | 137.305 | -26.420 | -1.479 | 120.641 |
| Main400/q2.8 | BestLayer − NativeStrong | 3 | -13713.916 | -227.113 | 0.026 | 584.114 | 188.884 | 0.066 | -13167.940 |
| Main400/q2.8 | BestLayer − NativeCommon1 | 3 | -13722.803 | -227.101 | -0.146 | 504.749 | 186.920 | -2.653 | -13261.035 |
| Main400/q2.8 | BestToken − NativeStrong | 3 | -17106.287 | -244.895 | 1.152 | 451.869 | 190.762 | 1.506 | -16705.893 |
| Main400/q2.8 | BestToken − NativeCommon2 | 3 | -17155.896 | -245.264 | 0.713 | 376.529 | 188.410 | 1.447 | -16834.060 |
| Main400/q2.8 | BestHybrid − NativeStrong | 3 | -15124.993 | -222.578 | 0.119 | 569.705 | 321.895 | 0.018 | -14455.833 |
| Main400/q2.8 | BestHybrid − NativeCommon2 | 3 | -15174.603 | -222.946 | -0.320 | 494.365 | 319.544 | -0.041 | -14584.001 |
| Main400/q2.8 | TokenCap2Control − NativeStrong | 3 | -17088.872 | -250.057 | 0.134 | 589.175 | 164.342 | 0.027 | -16585.252 |
| Main400/q2.8 | TokenCap2Control − NativeCommon1 | 3 | -17097.759 | -250.045 | -0.038 | 509.810 | 162.378 | -2.691 | -16678.346 |
| Main400/q4.5 | BestLayer − BestToken | 1 | 6610.725 | -29.104 | -0.028 | 35.877 | -28.266 | -0.019 | 6589.185 |
| Main400/q4.5 | BestHybrid − BestToken | 1 | 4602.370 | -17.899 | 0.036 | 62.836 | 123.650 | -0.049 | 4770.944 |
| Main400/q4.5 | BestHybrid − BestLayer | 1 | -2008.355 | 11.206 | 0.064 | 26.958 | 151.916 | -0.030 | -1818.241 |
| Main400/q4.5 | TokenCap2Control − BestToken | 1 | 122.837 | -11.075 | 0.014 | 166.944 | -43.002 | -0.036 | 235.683 |
| Main400/q4.5 | BestLayer − NativeStrong | 1 | -16557.485 | -210.239 | 0.001 | 616.729 | 198.262 | 0.042 | -15952.690 |
| Main400/q4.5 | BestLayer − NativeCommon1 | 1 | -16649.547 | -210.419 | -0.136 | 535.487 | 197.367 | 0.043 | -16127.205 |
| Main400/q4.5 | BestToken − NativeStrong | 1 | -23168.210 | -181.135 | 0.029 | 580.852 | 226.529 | 0.062 | -22541.875 |
| Main400/q4.5 | BestToken − NativeCommon2 | 1 | -23202.982 | -180.583 | 0.465 | 499.906 | 223.425 | 0.103 | -22659.666 |
| Main400/q4.5 | BestHybrid − NativeStrong | 1 | -18565.840 | -199.034 | 0.065 | 643.688 | 350.178 | 0.013 | -17770.931 |
| Main400/q4.5 | BestHybrid − NativeCommon2 | 1 | -18600.612 | -198.481 | 0.501 | 562.741 | 347.075 | 0.054 | -17888.722 |
| Main400/q4.5 | TokenCap2Control − NativeStrong | 1 | -23045.373 | -192.210 | 0.043 | 747.796 | 183.527 | 0.026 | -22306.192 |
| Main400/q4.5 | TokenCap2Control − NativeCommon1 | 1 | -23137.435 | -192.390 | -0.093 | 666.554 | 182.632 | 0.026 | -22480.707 |

## Whole-prefill Deflection의 다른 경로

D-local Request에는 P 실행과 P→D Handoff가 없다. 이 단계를 0ms로 만들어 넣지 않는다.
다음 값은 선택된 Route 안의 조건부 Mean TTFT 차이다. Reference도 같은 Request ID만 사용한다.
Route 선택은 무작위가 아니다. 서로 다른 Route의 값을 하나의 P Service 인과 효과로 해석하지 않는다.

| Workload / QPS | Reference | Seed | Route | Requests | Mean TTFT 차이(ms) |
|---|---|---:|---|---:|---:|
| Heldout200/q1.8 | NativeStrong | 2026112411 | P_FULL36_THEN_D | 145 | -3682.926 |
| Heldout200/q1.8 | NativeStrong | 2026112411 | D_LOCAL_FULL36 | 55 | -4829.511 |
| Heldout200/q1.8 | NativeStrong | 2026112412 | P_FULL36_THEN_D | 147 | -2346.520 |
| Heldout200/q1.8 | NativeStrong | 2026112412 | D_LOCAL_FULL36 | 53 | -4350.931 |
| Heldout200/q1.8 | NativeStrong | 2026112413 | P_FULL36_THEN_D | 139 | -6446.427 |
| Heldout200/q1.8 | NativeStrong | 2026112413 | D_LOCAL_FULL36 | 61 | -7724.167 |
| Heldout200/q1.8 | NativeCommon1 | 2026112411 | P_FULL36_THEN_D | 145 | -3786.329 |
| Heldout200/q1.8 | NativeCommon1 | 2026112411 | D_LOCAL_FULL36 | 55 | -4848.010 |
| Heldout200/q1.8 | NativeCommon1 | 2026112412 | P_FULL36_THEN_D | 147 | -2457.446 |
| Heldout200/q1.8 | NativeCommon1 | 2026112412 | D_LOCAL_FULL36 | 53 | -4372.704 |
| Heldout200/q1.8 | NativeCommon1 | 2026112413 | P_FULL36_THEN_D | 139 | -6595.518 |
| Heldout200/q1.8 | NativeCommon1 | 2026112413 | D_LOCAL_FULL36 | 61 | -7795.215 |
| Heldout200/q2.8 | NativeStrong | 2026112411 | P_FULL36_THEN_D | 141 | -13549.564 |
| Heldout200/q2.8 | NativeStrong | 2026112411 | D_LOCAL_FULL36 | 59 | -17159.169 |
| Heldout200/q2.8 | NativeStrong | 2026112412 | P_FULL36_THEN_D | 136 | -14353.935 |
| Heldout200/q2.8 | NativeStrong | 2026112412 | D_LOCAL_FULL36 | 64 | -16257.212 |
| Heldout200/q2.8 | NativeStrong | 2026112413 | P_FULL36_THEN_D | 138 | -14046.730 |
| Heldout200/q2.8 | NativeStrong | 2026112413 | D_LOCAL_FULL36 | 62 | -16397.653 |
| Heldout200/q2.8 | NativeCommon0 | 2026112411 | P_FULL36_THEN_D | 141 | -13670.817 |
| Heldout200/q2.8 | NativeCommon0 | 2026112411 | D_LOCAL_FULL36 | 59 | -17278.677 |
| Heldout200/q2.8 | NativeCommon0 | 2026112412 | P_FULL36_THEN_D | 136 | -14414.082 |
| Heldout200/q2.8 | NativeCommon0 | 2026112412 | D_LOCAL_FULL36 | 64 | -16354.979 |
| Heldout200/q2.8 | NativeCommon0 | 2026112413 | P_FULL36_THEN_D | 138 | -14131.642 |
| Heldout200/q2.8 | NativeCommon0 | 2026112413 | D_LOCAL_FULL36 | 62 | -16456.727 |
| Main400/q1.0 | NativeStrong | 2026112411 | P_FULL36_THEN_D | 312 | -1.591 |
| Main400/q1.0 | NativeStrong | 2026112411 | D_LOCAL_FULL36 | 88 | -595.843 |
| Main400/q1.0 | NativeCommon1 | 2026112411 | P_FULL36_THEN_D | 312 | -75.092 |
| Main400/q1.0 | NativeCommon1 | 2026112411 | D_LOCAL_FULL36 | 88 | -671.014 |
| Main400/q1.8 | NativeStrong | 2026112411 | P_FULL36_THEN_D | 311 | -195.055 |
| Main400/q1.8 | NativeStrong | 2026112411 | D_LOCAL_FULL36 | 89 | -1304.832 |
| Main400/q1.8 | NativeStrong | 2026112412 | P_FULL36_THEN_D | 310 | -717.867 |
| Main400/q1.8 | NativeStrong | 2026112412 | D_LOCAL_FULL36 | 90 | -1672.047 |
| Main400/q1.8 | NativeStrong | 2026112413 | P_FULL36_THEN_D | 306 | -343.943 |
| Main400/q1.8 | NativeStrong | 2026112413 | D_LOCAL_FULL36 | 94 | -1331.944 |
| Main400/q1.8 | NativeCommon1 | 2026112411 | P_FULL36_THEN_D | 311 | -405.204 |
| Main400/q1.8 | NativeCommon1 | 2026112411 | D_LOCAL_FULL36 | 89 | -1443.926 |
| Main400/q1.8 | NativeCommon1 | 2026112412 | P_FULL36_THEN_D | 310 | -850.379 |
| Main400/q1.8 | NativeCommon1 | 2026112412 | D_LOCAL_FULL36 | 90 | -1753.660 |
| Main400/q1.8 | NativeCommon1 | 2026112413 | P_FULL36_THEN_D | 306 | -487.792 |
| Main400/q1.8 | NativeCommon1 | 2026112413 | D_LOCAL_FULL36 | 94 | -1446.930 |
| Main400/q2.8 | NativeStrong | 2026112411 | P_FULL36_THEN_D | 297 | -16101.059 |
| Main400/q2.8 | NativeStrong | 2026112411 | D_LOCAL_FULL36 | 103 | -18629.967 |
| Main400/q2.8 | NativeStrong | 2026112412 | P_FULL36_THEN_D | 303 | -12408.433 |
| Main400/q2.8 | NativeStrong | 2026112412 | D_LOCAL_FULL36 | 97 | -13825.840 |
| Main400/q2.8 | NativeStrong | 2026112413 | P_FULL36_THEN_D | 293 | -18067.682 |
| Main400/q2.8 | NativeStrong | 2026112413 | D_LOCAL_FULL36 | 107 | -19784.634 |
| Main400/q2.8 | NativeCommon0 | 2026112411 | P_FULL36_THEN_D | 297 | -16295.171 |
| Main400/q2.8 | NativeCommon0 | 2026112411 | D_LOCAL_FULL36 | 103 | -18727.504 |
| Main400/q2.8 | NativeCommon0 | 2026112412 | P_FULL36_THEN_D | 303 | -12753.405 |
| Main400/q2.8 | NativeCommon0 | 2026112412 | D_LOCAL_FULL36 | 97 | -14130.667 |
| Main400/q2.8 | NativeCommon0 | 2026112413 | P_FULL36_THEN_D | 293 | -18493.665 |
| Main400/q2.8 | NativeCommon0 | 2026112413 | D_LOCAL_FULL36 | 107 | -20137.169 |
| Main400/q4.5 | NativeStrong | 2026112411 | P_FULL36_THEN_D | 294 | -19841.550 |
| Main400/q4.5 | NativeStrong | 2026112411 | D_LOCAL_FULL36 | 106 | -21029.748 |
| Main400/q4.5 | NativeCommon0 | 2026112411 | P_FULL36_THEN_D | 294 | -20073.621 |
| Main400/q4.5 | NativeCommon0 | 2026112411 | D_LOCAL_FULL36 | 106 | -21220.526 |

각 Stage의 절대 차이 CI, 실제 Output 수, Credit와 Compile Family는 전체 JSON에 있다.
N1에는 독립 Start CI가 없다. N3 CI는 Pointwise Empirical Bootstrap 27개 재표본이다.
첫 Output 이후 Decode Residency는 TTFT와 분리해 기록한다.
Host 구간의 증가를 실제 GPU Attention 비용의 증가로 자동 해석하지 않는다.

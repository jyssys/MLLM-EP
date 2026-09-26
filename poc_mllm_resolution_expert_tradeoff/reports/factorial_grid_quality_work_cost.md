
# Factorial grid: quality, work, and physical cost

Evidence is the public-partial official generation lane on actual TRUE EP4.
Discovery generation timings are not used as clean performance evidence; the
table uses the separate paired representative timing cohort.

| R | actual V | K_v | score % by task | visual pairs | physical rows | max/mean | EP critical ms | TTFT ms | natural E2E ms | mean output tokens | normal stop % | generation timeout | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 398.6 | 8 | {"chartqa": 84.38, "mmstar": 70.31} | 153056 | N/A | 1.159 | 78.072 | 3215.621 | 4517.555 | 166.3 | 100.00 | 0 | ACTUAL_TRUE_EP4 |
| R0 | 398.6 | 6 | {"chartqa": 84.38, "mmstar": 71.88} | 114792 | N/A | 1.177 | 84.021 | 3540.097 | 7370.770 | 134.8 | 100.00 | 0 | ACTUAL_TRUE_EP4 |
| R0 | 398.6 | 4 | {"chartqa": 87.5, "mmstar": 67.19} | 76528 | N/A | 1.226 | 83.527 | 3528.360 | 30168.845 | 126.1 | 100.00 | 0 | ACTUAL_TRUE_EP4 |
| R05 | 194.8 | 8 | {"chartqa": 90.62, "mmstar": 68.75} | 74820 | N/A | 1.197 | 83.691 | 3195.219 | 4549.162 | 212.5 | 100.00 | 0 | ACTUAL_TRUE_EP4 |
| R05 | 194.8 | 6 | {"chartqa": 90.62, "mmstar": 67.19} | 56115 | N/A | 1.211 | 85.753 | 3421.733 | 7397.797 | 145.1 | 100.00 | 0 | ACTUAL_TRUE_EP4 |
| R05 | 194.8 | 4 | {"chartqa": 87.5, "mmstar": 65.62} | 37410 | N/A | 1.219 | 84.078 | 3506.708 | 8217.186 | 147.9 | 100.00 | 0 | ACTUAL_TRUE_EP4 |
| R025 | 105.6 | 8 | {"chartqa": 78.12, "mmstar": 67.19} | 40536 | N/A | 1.210 | 77.240 | 3160.670 | 4422.446 | 150.8 | 100.00 | 0 | ACTUAL_TRUE_EP4 |
| R025 | 105.6 | 6 | {"chartqa": 75.0, "mmstar": 68.75} | 30402 | N/A | 1.210 | 81.716 | 3429.327 | 7344.595 | 134.8 | 100.00 | 0 | ACTUAL_TRUE_EP4 |
| R025 | 105.6 | 4 | {"chartqa": 75.0, "mmstar": 60.94} | 20268 | N/A | 1.226 | 82.581 | 3412.446 | 18296.404 | 138.0 | 100.00 | 0 | ACTUAL_TRUE_EP4 |

The physical-row column remains `N/A` when the backend proof exposes only
logical compact routes and not kernel padding. It is never inferred from the
fixed-width tensor slots.

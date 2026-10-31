# Two-GPU W1–W4: Measured Seven-Arm Matrix

Only COMPLETE cells passed the recorded output and work checks. Other status values remain explicit.

## ouro / W1

| Arm | Status | Physical GPUs | Completed/offered | Normal arrival intensity req/s | Joint SLO % | Goodput req/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | ITL p99 ms | E2E p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | COMPLETE | [0, 1] | 32/32 | 0.2118 | 100.00 | 0.20042 | 286.88 | 31.75 | 33.20 | 33.90 | 15411.23 |
| DP_Chunked | COMPLETE | [0, 1] | 32/32 | 0.2118 | 100.00 | 0.19852 | 2207.12 | 34.94 | 33.91 | 103.08 | 17211.35 |
| PD_G1 | COMPLETE | [0, 1] | 32/32 | 0.2118 | 100.00 | 0.19870 | 3449.61 | 27.98 | 28.05 | 33.28 | 16086.66 |
| Loop_Static | COMPLETE | [0, 1] | 32/32 | 0.2118 | 100.00 | 0.19823 | 3469.37 | 27.45 | 27.44 | 30.66 | 16107.21 |
| Loop_Dynamic | COMPLETE | [0, 1] | 32/32 | 0.2118 | 100.00 | 0.19883 | 4369.44 | 34.01 | 34.36 | 38.57 | 18204.44 |
| Token_Static | COMPLETE | [0, 1] | 32/32 | 0.2118 | 100.00 | 0.19841 | 7503.65 | 34.67 | 34.44 | 52.79 | 21896.34 |
| Token_Dynamic | COMPLETE | [0, 1] | 32/32 | 0.2118 | 100.00 | 0.19891 | 3241.92 | 31.84 | 33.52 | 34.07 | 16705.75 |

## ouro / W2

| Arm | Status | Physical GPUs | Completed/offered | Normal arrival intensity req/s | Joint SLO % | Goodput req/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | ITL p99 ms | E2E p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | COMPLETE | [2, 3] | 24/24 | 0.3326 | 100.00 | 0.36562 | 294.03 | 32.42 | 32.72 | 33.66 | 7483.18 |
| DP_Chunked | COMPLETE | [2, 3] | 24/24 | 0.3326 | 83.33 | 0.29778 | 3470.28 | 48.00 | 99.82 | 108.07 | 11760.13 |
| PD_G1 | COMPLETE | [2, 3] | 24/24 | 0.3326 | 100.00 | 0.33995 | 11604.75 | 33.69 | 34.28 | 37.35 | 17822.16 |
| Loop_Static | COMPLETE | [2, 3] | 24/24 | 0.3326 | 54.17 | 0.18138 | 18297.65 | 52.82 | 105.84 | 112.20 | 29620.92 |
| Loop_Dynamic | COMPLETE | [2, 3] | 24/24 | 0.3326 | 100.00 | 0.34844 | 6537.23 | 35.33 | 33.40 | 47.48 | 11844.25 |
| Token_Static | COMPLETE | [2, 3] | 24/24 | 0.3326 | 100.00 | 0.33595 | 10446.74 | 35.22 | 34.57 | 88.05 | 18152.72 |
| Token_Dynamic | COMPLETE | [2, 3] | 24/24 | 0.3326 | 100.00 | 0.35195 | 5686.03 | 34.13 | 32.24 | 37.90 | 10981.24 |

## ouro / W3

| Arm | Status | Physical GPUs | Completed/offered | Normal arrival intensity req/s | Joint SLO % | Goodput req/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | ITL p99 ms | E2E p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | COMPLETE | [4, 5] | 32/32 | 0.2756 | 96.88 | 0.33058 | 18242.25 | 38.17 | 36.25 | 51.78 | 29228.17 |
| DP_Chunked | COMPLETE | [4, 5] | 32/32 | 0.2756 | 71.88 | 0.22516 | 22616.39 | 42.21 | 95.60 | 103.17 | 33507.61 |
| PD_G1 | COMPLETE | [4, 5] | 32/32 | 0.2756 | 56.25 | 0.15582 | 39970.06 | 32.01 | 32.87 | 35.19 | 49359.89 |
| Loop_Static | COMPLETE | [4, 5] | 32/32 | 0.2756 | 56.25 | 0.15481 | 41187.34 | 33.34 | 34.62 | 41.83 | 50596.38 |
| Loop_Dynamic | COMPLETE | [4, 5] | 32/32 | 0.2756 | 50.00 | 0.12131 | 52346.13 | 35.00 | 35.45 | 38.19 | 65701.20 |
| Token_Static | COMPLETE | [4, 5] | 32/32 | 0.2756 | 34.38 | 0.08169 | 59969.11 | 41.83 | 85.43 | 103.02 | 68776.89 |
| Token_Dynamic | COMPLETE | [4, 5] | 32/32 | 0.2756 | 68.75 | 0.19763 | 36591.49 | 27.63 | 28.03 | 35.41 | 46274.93 |

## ouro / W4

| Arm | Status | Physical GPUs | Completed/offered | Normal arrival intensity req/s | Joint SLO % | Goodput req/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | ITL p99 ms | E2E p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | COMPLETE | [6, 7] | 24/24 | 0.2728 | 100.00 | 0.27179 | 190.90 | 27.41 | 30.05 | 31.81 | 13754.93 |
| DP_Chunked | COMPLETE | [6, 7] | 24/24 | 0.2728 | 100.00 | 0.26694 | 1168.37 | 36.27 | 32.43 | 159.94 | 19155.05 |
| PD_G1 | COMPLETE | [6, 7] | 24/24 | 0.2728 | 100.00 | 0.26960 | 3944.33 | 29.84 | 30.56 | 31.19 | 15781.72 |
| Loop_Static | COMPLETE | [6, 7] | 24/24 | 0.2728 | 100.00 | 0.26977 | 8087.91 | 33.53 | 31.97 | 32.94 | 18996.26 |
| Loop_Dynamic | COMPLETE | [6, 7] | 24/24 | 0.2728 | 100.00 | 0.25533 | 7209.25 | 31.53 | 31.96 | 36.71 | 21179.71 |
| Token_Static | COMPLETE | [6, 7] | 24/24 | 0.2728 | 100.00 | 0.24703 | 7594.01 | 32.58 | 32.22 | 70.09 | 21927.37 |
| Token_Dynamic | COMPLETE | [6, 7] | 24/24 | 0.2728 | 100.00 | 0.24931 | 7116.91 | 32.07 | 32.54 | 35.00 | 21059.50 |

## huginn / W1

| Arm | Status | Physical GPUs | Completed/offered | Normal arrival intensity req/s | Joint SLO % | Goodput req/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | ITL p99 ms | E2E p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | COMPLETE | [4, 5] | 32/32 | 0.0527 | 100.00 | 0.05006 | 1432.52 | 107.25 | 104.78 | 110.93 | 53330.82 |
| DP_Chunked | COMPLETE | [4, 5] | 32/32 | 0.0527 | 100.00 | 0.04997 | 4019.38 | 107.33 | 104.44 | 261.68 | 55481.69 |
| PD_G1 | COMPLETE | [4, 5] | 32/32 | 0.0527 | 100.00 | 0.04962 | 37659.22 | 131.50 | 135.06 | 135.96 | 95299.85 |
| Loop_Static | COMPLETE | [4, 5] | 32/32 | 0.0527 | 100.00 | 0.04994 | 23161.95 | 117.36 | 122.21 | 123.65 | 73649.31 |
| Loop_Dynamic | COMPLETE | [4, 5] | 32/32 | 0.0527 | 100.00 | 0.04994 | 24051.56 | 117.72 | 121.67 | 122.60 | 76613.90 |
| Token_Static | COMPLETE | [4, 5] | 32/32 | 0.0527 | 100.00 | 0.04985 | 29830.89 | 120.63 | 122.68 | 261.64 | 85506.78 |
| Token_Dynamic | COMPLETE | [4, 5] | 32/32 | 0.0527 | 100.00 | 0.04991 | 22751.88 | 115.33 | 117.99 | 125.61 | 75195.53 |

## huginn / W2

| Arm | Status | Physical GPUs | Completed/offered | Normal arrival intensity req/s | Joint SLO % | Goodput req/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | ITL p99 ms | E2E p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | COMPLETE | [6, 7] | 24/24 | 0.0794 | 100.00 | 0.08742 | 1336.09 | 112.74 | 108.04 | 113.22 | 29125.21 |
| DP_Chunked | COMPLETE | [6, 7] | 24/24 | 0.0794 | 100.00 | 0.08676 | 3887.26 | 125.40 | 114.62 | 279.74 | 31445.66 |
| PD_G1 | COMPLETE | [6, 7] | 24/24 | 0.0794 | 100.00 | 0.08111 | 71662.77 | 119.93 | 121.99 | 122.62 | 101023.02 |
| Loop_Static | COMPLETE | [6, 7] | 24/24 | 0.0794 | 100.00 | 0.08221 | 65154.81 | 115.00 | 118.19 | 122.22 | 92781.30 |
| Loop_Dynamic | COMPLETE | [6, 7] | 24/24 | 0.0794 | 100.00 | 0.08143 | 75929.45 | 123.55 | 123.34 | 129.74 | 105352.03 |
| Token_Static | COMPLETE | [6, 7] | 24/24 | 0.0794 | 100.00 | 0.08205 | 76696.53 | 130.18 | 130.28 | 203.14 | 106473.26 |
| Token_Dynamic | COMPLETE | [6, 7] | 24/24 | 0.0794 | 100.00 | 0.08243 | 64605.86 | 114.69 | 118.21 | 118.99 | 92139.14 |

## huginn / W3

| Arm | Status | Physical GPUs | Completed/offered | Normal arrival intensity req/s | Joint SLO % | Goodput req/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | ITL p99 ms | E2E p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | COMPLETE | [0, 1] | 32/32 | 0.0628 | 93.75 | 0.07927 | 67541.51 | 123.37 | 120.42 | 125.33 | 116915.73 |
| DP_Chunked | COMPLETE | [0, 1] | 32/32 | 0.0628 | 93.75 | 0.07758 | 83882.58 | 131.29 | 125.49 | 294.05 | 136243.39 |
| PD_G1 | COMPLETE | [0, 1] | 32/32 | 0.0628 | 62.50 | 0.03718 | 245435.16 | 120.80 | 123.77 | 128.15 | 278563.40 |
| Loop_Static | COMPLETE | [0, 1] | 32/32 | 0.0628 | 68.75 | 0.04033 | 258503.63 | 125.29 | 126.04 | 265.25 | 298312.23 |
| Loop_Dynamic | COMPLETE | [0, 1] | 32/32 | 0.0628 | 62.50 | 0.03768 | 243088.23 | 122.56 | 123.98 | 128.60 | 276363.02 |
| Token_Static | COMPLETE | [0, 1] | 32/32 | 0.0628 | 65.62 | 0.03604 | 282906.32 | 132.20 | 134.10 | 150.34 | 335015.01 |
| Token_Dynamic | COMPLETE | [0, 1] | 32/32 | 0.0628 | 68.75 | 0.03999 | 254166.96 | 121.38 | 124.84 | 128.65 | 303918.31 |

## huginn / W4

| Arm | Status | Physical GPUs | Completed/offered | Normal arrival intensity req/s | Joint SLO % | Goodput req/s | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | ITL p99 ms | E2E p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | COMPLETE | [2, 3] | 24/24 | 0.0797 | 100.00 | 0.07805 | 491.46 | 107.43 | 110.25 | 113.95 | 55201.12 |
| DP_Chunked | COMPLETE | [2, 3] | 24/24 | 0.0797 | 100.00 | 0.07784 | 1618.74 | 113.43 | 110.69 | 270.97 | 57287.39 |
| PD_G1 | COMPLETE | [2, 3] | 24/24 | 0.0797 | 100.00 | 0.06357 | 58928.99 | 131.47 | 131.99 | 133.73 | 118543.29 |
| Loop_Static | COMPLETE | [2, 3] | 24/24 | 0.0797 | 100.00 | 0.06644 | 46649.77 | 124.22 | 128.28 | 290.04 | 103109.16 |
| Loop_Dynamic | COMPLETE | [2, 3] | 24/24 | 0.0797 | 100.00 | 0.07028 | 30304.32 | 117.35 | 118.09 | 119.28 | 81796.72 |
| Token_Static | COMPLETE | [2, 3] | 24/24 | 0.0797 | 100.00 | 0.06808 | 42176.49 | 120.78 | 119.41 | 235.61 | 93431.40 |
| Token_Dynamic | COMPLETE | [2, 3] | 24/24 | 0.0797 | 100.00 | 0.06709 | 39748.58 | 123.63 | 124.22 | 127.00 | 95609.68 |

## Full metric and uncertainty tables

The CSV includes request-latency mean/p50/p95/p99, throughput, memory, communication, failures, and depth accounting.
ITL quantiles use observed single-token delivery intervals. Multi-token events remain counted separately.
One primary start does not provide a qualifying independent-start confidence interval.
Displayed values are rounded. The CSV files retain unrounded values.
The paired table separates each repeated 1.5-rate W1 comparison.

| Model | Baseline | Paired starts | Relative goodput gain % | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| ouro | PD_G1 | 3 | 1.95 | -3.39 | 9.10 |
| ouro | DP | 3 | -7.46 | -16.63 | -0.41 |
| ouro | Loop_Static | 3 | 1.48 | 0.66 | 1.98 |
| ouro | Token_Dynamic | 3 | 0.30 | -2.69 | 3.48 |
| huginn | PD_G1 | 3 | 7.70 | -0.51 | 16.52 |
| huginn | DP | 3 | -25.85 | -35.38 | -15.46 |
| huginn | Loop_Static | 3 | 0.22 | -0.41 | 1.11 |
| huginn | Token_Dynamic | 3 | 4.45 | 1.55 | 9.68 |

# Separate heavy quantum tables

Evidence: HEAVY_MEASURED, not clean serving latency.
The stream interval includes staging, stream waits, and host launch gaps.
It is not pure GPU compute or removable E2E time.
Each shape/policy normally has one profile process. Submission counts are not independent samples.
These rows select the new long prompt only. Background priming quanta remain in the full CSV.
Full-C rows exclude the short final prompt chunk when the observed row count permits that filter.
Mixed can combine decode and prefill rows. It is not a pure-G control.

| P/N | Policy | C | G | GPU | Submissions | Mean/P95 stream ms | Full-C mean ms | Mean prompt/decode rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048/4 | atomic | 128 | 4 | [6] | 16 | 76.29/77.03 | 76.27 | 127.06/0.00 |
| 2048/4 | atomic | 256 | 4 | [6] | 8 | 79.62/80.01 | 79.55 | 254.12/0.00 |
| 2048/4 | atomic | 512 | 4 | [6] | 4 | 87.39/87.74 | 87.46 | 508.25/0.00 |
| 2048/4 | atomic | 1024 | 4 | [6] | 2 | 103.33/104.43 | 102.11 | 1016.50/0.00 |
| 2048/4 | grouped | 128 | 2 | [5] | 32 | 38.73/40.18 | 38.81 | 127.06/0.00 |
| 2048/4 | grouped | 256 | 2 | [5] | 16 | 40.19/40.80 | 40.18 | 254.12/0.00 |
| 2048/4 | grouped | 512 | 2 | [5] | 8 | 44.40/47.50 | 43.88 | 508.25/0.00 |
| 2048/4 | grouped | 1024 | 2 | [5] | 4 | 50.64/50.81 | 50.77 | 1016.50/0.00 |
| 2048/4 | mixed | 128 | 1 | [7] | 64 | 21.04/21.38 | 21.03 | 127.06/3.00 |
| 2048/4 | mixed | 256 | 1 | [7] | 32 | 22.89/29.43 | 22.99 | 254.12/3.00 |
| 2048/4 | mixed | 512 | 1 | [7] | 16 | 26.15/33.96 | 26.10 | 508.25/3.00 |
| 2048/4 | mixed | 1024 | 1 | [7] | 12 | 25.61/29.03 | NOT_AVAILABLE | 677.67/3.00 |
| 2048/4 | split | 128 | 1 | [4] | 64 | 20.21/20.73 | 20.22 | 127.06/0.00 |
| 2048/4 | split | 256 | 1 | [4] | 32 | 21.12/21.55 | 21.13 | 254.12/0.00 |
| 2048/4 | split | 512 | 1 | [4] | 16 | 22.88/23.70 | 22.84 | 508.25/0.00 |
| 2048/4 | split | 1024 | 1 | [4] | 8 | 26.92/27.85 | 26.68 | 1016.50/0.00 |
| 8192/8 | atomic | 128 | 4 | [4] | 64 | 78.70/79.09 | 78.70 | 127.77/0.00 |
| 8192/8 | atomic | 256 | 4 | [6] | 32 | 79.22/79.72 | 79.21 | 255.53/0.00 |
| 8192/8 | atomic | 512 | 4 | [6] | 16 | 86.66/87.21 | 86.67 | 511.06/0.00 |
| 8192/8 | atomic | 1024 | 4 | [4] | 8 | 102.72/103.34 | 102.83 | 1022.12/0.00 |
| 8192/8 | grouped | 128 | 2 | [6] | 128 | 39.60/41.98 | 39.59 | 127.77/0.00 |
| 8192/8 | grouped | 256 | 2 | [7] | 64 | 44.15/44.80 | 44.17 | 255.53/0.00 |
| 8192/8 | grouped | 512 | 2 | [7] | 32 | 48.15/52.37 | 48.20 | 511.06/0.00 |
| 8192/8 | grouped | 1024 | 2 | [6] | 16 | 52.11/52.51 | 52.09 | 1022.12/0.00 |
| 8192/8 | mixed | 128 | 1 | [7] | 256 | 20.81/21.22 | 20.81 | 127.77/7.00 |
| 8192/8 | mixed | 256 | 1 | [4] | 128 | 21.74/22.00 | 21.73 | 255.53/7.00 |
| 8192/8 | mixed | 512 | 1 | [4] | 64 | 23.71/24.02 | 23.70 | 511.06/7.00 |
| 8192/8 | mixed | 1024 | 1 | [7] | 60 | 24.30/28.42 | NOT_AVAILABLE | 545.13/7.00 |
| 8192/8 | split | 128 | 1 | [5] | 256 | 20.52/20.84 | 20.52 | 127.77/0.00 |
| 8192/8 | split | 256 | 1 | [5] | 128 | 21.31/21.34 | 21.31 | 255.53/0.00 |
| 8192/8 | split | 512 | 1 | [5] | 64 | 23.04/23.31 | 22.98 | 511.06/0.00 |
| 8192/8 | split | 1024 | 1 | [5] | 32 | 27.44/28.01 | 27.43 | 1022.12/0.00 |
| 8192/8 | staged4 | 1024 | 4 | [4, 5, 6, 7] | 32 | 103.25/105.93 | 103.37 | 1022.12/0.00 |

The long-prompt subset is `long_quantum_response_surface.csv`.
Prelude, background priming, and decode/coda records remain separate in `quantum_response_surface.csv`.
No held-out online quantum-cost predictor is validated by this table.

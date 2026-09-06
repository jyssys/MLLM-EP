# E2E latency mass atlas (fresh live traces)

All rows below exclude warmup when the measured atlas exists. The stage mass is the sum of rank-collapsed, layer-local CUDA intervals; it is not treated as request-level E2E time because request IDs are only wave labels.

| trace | phase | n | M p50 | T_MoE p50/p90/p99 ms | dispatch % | expert % | combine % | wait % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| live_alternating_text_shapes_c8 | decode | 768 | 1 | 1.215/1.513/2.056 | 16.7 | 35.3 | 3.6 | 8.0 |
| live_alternating_text_shapes_c8 | prefill | 2304 | 199 | 1.252/1.695/3.880 | 18.7 | 37.2 | 3.0 | 8.7 |
| live_alternating_text_vision_c8 | decode | 768 | 1 | 2.121/2.187/2.809 | 34.7 | 30.8 | 3.3 | 24.1 |
| live_alternating_text_vision_c8 | prefill | 2304 | 154 | 2.118/2.260/3.816 | 29.7 | 33.9 | 3.4 | 24.2 |
| live_alternating_vision_shapes_c8 | decode | 768 | 1 | 1.141/1.223/1.990 | 10.9 | 38.5 | 3.9 | 8.0 |
| live_alternating_vision_shapes_c8 | prefill | 2304 | 165 | 1.161/1.248/1.963 | 11.2 | 42.7 | 2.8 | 6.8 |
| live_high_c16 | prefill | 3072 | 284 | 1.193/1.449/5.628 | 20.3 | 38.6 | 2.3 | 7.6 |
| live_long_text_c8 | prefill | 1536 | 568 | 2.012/5.986/6.166 | 45.9 | 25.4 | 1.6 | 13.1 |
| live_mixed_c2 | prefill | 2304 | 165 | 1.209/1.984/3.791 | 29.3 | 33.1 | 3.3 | 12.1 |
| live_mixed_c4 | decode | 1392 | 1 | 1.248/2.850/7.693 | 28.5 | 31.5 | 3.2 | 11.3 |
| live_mixed_c4 | prefill | 1488 | 193 | 1.271/2.307/8.369 | 27.5 | 32.3 | 3.0 | 9.9 |
| live_mixed_c8 | decode | 3840 | 1 | 1.151/1.782/3.941 | 33.3 | 28.5 | 3.0 | 8.4 |
| live_mixed_c8 | prefill | 8448 | 193 | 1.164/1.401/3.989 | 17.8 | 38.4 | 2.9 | 6.0 |
| live_mixed_c8_mbt4096 | decode | 1440 | 1 | 1.943/2.357/2.675 | 38.6 | 25.5 | 2.6 | 15.0 |
| live_mixed_c8_mbt4096 | prefill | 3168 | 193 | 2.055/4.239/5.086 | 48.9 | 22.4 | 1.9 | 12.0 |
| live_text_c16 | prefill | 3072 | 456 | 1.399/2.030/3.781 | 28.4 | 34.9 | 2.5 | 14.6 |
| live_text_c4 | prefill | 1488 | 114 | 1.171/1.440/6.236 | 21.1 | 33.8 | 3.3 | 8.9 |
| live_text_c8 | prefill | 3840 | 228 | 1.168/1.269/3.326 | 30.4 | 31.0 | 3.0 | 3.3 |
| live_text_c8_rep2 | prefill | 2160 | 342 | 2.056/4.205/7.021 | 41.1 | 24.9 | 2.4 | 16.1 |
| live_text_c8_telemetry | decode | 3840 | 1 | 2.071/2.890/3.971 | 39.4 | 24.1 | 2.6 | 19.7 |
| live_text_c8_telemetry | prefill | 11520 | 114 | 2.053/4.094/4.224 | 43.3 | 22.8 | 2.4 | 16.6 |
| live_text_c8_warmup_text | decode | 768 | 1 | 1.161/1.249/1.991 | 12.5 | 37.1 | 3.9 | 7.5 |
| live_text_c8_warmup_text | prefill | 2304 | 114 | 1.171/1.258/2.417 | 12.3 | 38.8 | 3.9 | 5.4 |
| live_text_c8_warmup_text_long | decode | 3072 | 1 | 1.175/1.322/4.408 | 14.3 | 37.0 | 3.8 | 9.9 |
| live_text_c8_warmup_text_long | prefill | 9216 | 114 | 1.181/1.336/2.869 | 13.8 | 38.2 | 3.8 | 5.4 |
| live_text_to_vision_c8 | decode | 768 | 1 | 2.194/2.588/2.756 | 32.3 | 26.0 | 2.7 | 12.9 |
| live_text_to_vision_c8 | prefill | 2304 | 193 | 2.395/5.031/5.859 | 45.6 | 22.6 | 1.9 | 8.9 |
| live_text_to_vision_c8_telemetry | decode | 768 | 1 | 1.155/1.251/2.458 | 13.3 | 37.0 | 3.8 | 8.5 |
| live_text_to_vision_c8_telemetry | prefill | 2304 | 193 | 1.202/1.451/5.488 | 17.7 | 39.3 | 2.5 | 11.4 |
| live_vision_hi_c8 | prefill | 3840 | 386 | 1.218/1.338/4.470 | 15.1 | 42.2 | 2.0 | 7.9 |
| live_vision_hi_c8_warmup_hi | decode | 768 | 1 | 1.171/1.318/6.472 | 17.3 | 35.0 | 3.7 | 10.6 |
| live_vision_hi_c8_warmup_hi | prefill | 2304 | 193 | 1.204/1.309/1.971 | 12.1 | 41.8 | 2.7 | 6.7 |

## Direct request observations

- live_mixed_c2: n=48 E2E p50/p90/p99=285.63/390.41/1148.87 ms
- live_mixed_c4: n=32 E2E p50/p90/p99=811.49/1138.34/1543.91 ms
- live_mixed_c8: n=224 E2E p50/p90/p99=900.94/1115.47/6235.02 ms
- live_mixed_c8_mbt4096: n=84 E2E p50/p90/p99=1277.58/1865.11/2298.39 ms
- live_high_c16: n=144 E2E p50/p90/p99=790.93/1048.44/4127.09 ms
- live_long_text_c8: n=64 E2E p50/p90/p99=965.30/1583.39/1659.73 ms
- live_text_c4: n=32 E2E p50/p90/p99=443.42/806.76/892.29 ms
- live_text_c8: n=160 E2E p50/p90/p99=655.98/1030.10/1943.75 ms
- live_text_c8_rep2: n=96 E2E p50/p90/p99=969.99/1005.46/3363.84 ms
- live_text_c16: n=256 E2E p50/p90/p99=773.12/1759.07/2193.54 ms
- live_vision_hi_c8: n=160 E2E p50/p90/p99=757.82/1644.82/7617.81 ms
- live_text_c8_telemetry: n=320 E2E p50/p90/p99=1230.41/1309.05/1349.79 ms
- live_text_c8_warmup_text: n=64 E2E p50/p90/p99=842.66/854.65/857.45 ms
- live_text_to_vision_c8: n=64 E2E p50/p90/p99=1349.88/1541.25/4119.02 ms
- live_alternating_text_vision_c8: n=64 E2E p50/p90/p99=1295.72/1719.67/4200.41 ms
- live_alternating_text_shapes_c8: n=64 E2E p50/p90/p99=933.60/1169.86/1217.93 ms
- live_vision_hi_c8_warmup_hi: n=64 E2E p50/p90/p99=917.93/949.73/1336.55 ms
- live_alternating_vision_shapes_c8: n=64 E2E p50/p90/p99=860.91/943.60/1004.58 ms
- live_text_c8_warmup_text_long: n=256 E2E p50/p90/p99=841.61/943.10/1839.20 ms
- live_text_to_vision_c8_telemetry: n=64 E2E p50/p90/p99=951.91/2226.54/4029.00 ms

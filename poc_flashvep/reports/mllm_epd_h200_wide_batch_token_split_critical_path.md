# H200 wide-batch token split critical path

| Start | Method | QPS | Output | Observation | Arrival to proxy | Proxy to E submit | E HTTP | E complete to P submit | P HTTP | P ready to D submit | D submit to first client token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| online_cap_1 | native-common-B1 | 5.00 | Natural EOS | clean | 2.62 | 0.08 | 3051.53 | 0.15 | 2003.51 | 1512.34 | 60.17 |
| online_cap_16 | native-common-B16 | 5.00 | Natural EOS | clean | 3.32 | 0.04 | 3148.62 | 0.17 | 1737.64 | 0.24 | 73.75 |
| online_cap_2 | native-common-B2 | 5.00 | Natural EOS | clean | 2.58 | 0.04 | 3134.17 | 0.16 | 1769.74 | 43.45 | 67.88 |
| online_cap_32 | native-common-B32 | 5.00 | Natural EOS | clean | 2.79 | 0.05 | 2816.69 | 0.17 | 2555.13 | 0.23 | 77.38 |
| online_cap_4 | native-common-B4 | 5.00 | Natural EOS | clean | 2.70 | 0.04 | 3228.31 | 0.17 | 1795.50 | 5.57 | 71.72 |
| online_cap_64 | native-common-B64 | 5.00 | Natural EOS | clean | 2.96 | 0.04 | 3127.67 | 0.14 | 1741.07 | 0.20 | 67.04 |
| online_cap_8 | native-common-B8 | 5.00 | Natural EOS | clean | 2.78 | 0.04 | 3113.02 | 0.16 | 1723.76 | 0.22 | 71.28 |
| online_cap_free | native-common-free | 5.00 | Natural EOS | clean | 2.36 | 0.03 | 3062.73 | 0.16 | 2038.12 | 0.19 | 66.16 |
| pilot_b4_idle10 | idle10-B4 | 5.00 | Natural EOS | clean | 3.17 | 0.04 | 3361.71 | 0.17 | 2730.66 | 742.65 | 86.04 |
| pilot_b4_idle10 | idle10-B4 | 5.00 | Natural EOS | batch_observed | 2.96 | 0.05 | 3227.19 | 0.18 | 1673.36 | 484.44 | 91.68 |
| pilot_b4_native_common | native-common-B4 | 5.00 | Natural EOS | clean | 2.73 | 0.04 | 3196.53 | 0.14 | 3087.27 | 3.60 | 72.03 |
| pilot_b4_native_strong | native-strong-B4 | 5.00 | Natural EOS | clean | 2.83 | 0.05 | 3134.40 | 0.14 | 2816.23 | 3.47 | 68.47 |
| pilot_b4_static10 | static10-B4 | 5.00 | Natural EOS | clean | 3.26 | 0.08 | 3154.31 | 0.19 | 1406.16 | 4992.37 | 138.53 |
| pilot_b4_static10 | static10-B4 | 5.00 | Natural EOS | batch_observed | 2.51 | 0.04 | 2909.35 | 0.16 | 1732.28 | 4798.58 | 119.82 |
| pilot_free_idle10 | idle10-free | 5.00 | Natural EOS | clean | 3.27 | 0.05 | 3383.59 | 0.17 | 2732.29 | 320.20 | 88.96 |
| pilot_free_idle10 | idle10-free | 5.00 | Natural EOS | batch_observed | 2.75 | 0.08 | 3063.64 | 0.18 | 2121.85 | 451.36 | 86.93 |
| pilot_free_native_common | native-common-free | 5.00 | Natural EOS | clean | 3.17 | 0.04 | 3328.95 | 0.15 | 2947.76 | 0.18 | 70.23 |
| pilot_free_native_strong | native-strong-free | 5.00 | Natural EOS | clean | 3.23 | 0.04 | 3246.74 | 0.15 | 3107.49 | 0.17 | 67.75 |
| pilot_free_static10 | static10-free | 5.00 | Natural EOS | clean | 3.16 | 0.04 | 3168.41 | 0.16 | 1759.14 | 5320.52 | 129.75 |
| pilot_free_static10 | static10-free | 5.00 | Natural EOS | batch_observed | 3.26 | 0.04 | 3288.82 | 0.16 | 1299.29 | 4342.30 | 124.91 |

The seven columns partition scheduled arrival to first client output, for requests with monotonic timestamps
and complete proxy alignment. They are host clock boundaries in ms. The E and P HTTP spans include
submission, queues, transfer, execution, and response. The D submit to output boundary can include KV import,
scheduler, suffix, first worker emission, and client delivery. The mean along the common additive path can
be decomposed. Quantiles of individual stages do not add to TTFT p95.

For a split decision, check whether the drop in P HTTP/queue exceeds the extra D waiting and suffix work.
Compare the prepared decode distribution and D admission queue as well as ITL. A client-facing ITL gain
under a collapsed batch distribution is compatible with a loss in D capacity. Exact suffix kernel time and
a native-versus-split timeline require qualified GPU intervals aligned to the matching request boundary.

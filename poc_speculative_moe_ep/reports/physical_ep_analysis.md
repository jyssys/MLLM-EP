# Physical Ep Analysis

Measured clean TRUE EP4 on GPUs2,3,6,7, output cap64/EOS, first16 frozen requests (8+8). Route features come from a separate trace pass with identical request/prefix/draft IDs.

| Arm | acceptance | progress/verify | experts/layer | remote paths/layer | fanout/token | max-rank bytes/layer | EP ms/progress | target ms/progress | E2E ms/token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AR | N/A | 1.000 | 8.000 | 5.932 | 3.644 | 14924.6 | 25.953 | 90.804 | 92.779 |
| 2 | 0.858 | 2.673 | 17.123 | 17.846 | 3.642 | 29840.6 | 9.434 | 33.214 | 46.071 |
| 4 | 0.742 | 3.850 | 23.738 | 29.714 | 3.641 | 44742.9 | 7.140 | 23.832 | 41.043 |
| 8 | 0.626 | 5.712 | 33.822 | 53.530 | 3.643 | 74636.0 | 4.574 | 15.733 | 37.961 |

Remote path counts are token–expert paths; destination-deduplicated remote hidden rows and bytes are separate fields. Fanout includes the local destination. Per-rank matrices include useful local rows; subtract their diagonal for network-only logical attribution. Metadata/header/padding wire bytes are not inferred.

Cumulative-critical-rank D/E/C is one rank per invocation. Full-router projection is UNAVAILABLE independently; router selection only is measured. Dispatch/combine are prepare/finalize composites, not pure link service. Physical EP share of target: [(0, 0.28580948049430244), (2, 0.2840426656334929), (4, 0.2995736416150623), (8, 0.2907018293301598)].


Final key repeat check: **two predefined requests ×10 repeats/depth** (forward3+reverse7 restart blocks).

- gamma0: EP/progress=26.021ms; target/progress=90.153ms; E2E/token=92.130ms.
- gamma2: EP/progress=12.634ms; target/progress=36.386ms; E2E/token=51.363ms.
- gamma4: EP/progress=7.352ms; target/progress=26.045ms; E2E/token=45.221ms.
- gamma8: EP/progress=5.215ms; target/progress=18.406ms; E2E/token=45.182ms.

Instrumentation validity: {"clean_vs_bypass": [{"gamma": 0, "median_request_overhead_pct": 4.794913560328629, "request_values_pct": [4.539753668636637, 5.05007345202062], "status": "SMALL_SUBSET_COMPATIBLE", "scope": "separate restart blocks; not randomized interleave; two requests, medians of repeated actual completions"}, {"gamma": 2, "median_request_overhead_pct": 2.89607299608744, "request_values_pct": [2.745003498126164, 3.0471424940487157], "status": "SMALL_SUBSET_COMPATIBLE", "scope": "separate restart blocks; not randomized interleave; two requests, medians of repeated actual completions"}, {"gamma": 4, "median_request_overhead_pct": 5.558812690132631, "request_values_pct": [6.065780767109441, 5.05184461315582], "status": "SMALL_SUBSET_COMPATIBLE", "scope": "separate restart blocks; not randomized interleave; two requests, medians of repeated actual completions"}, {"gamma": 8, "median_request_overhead_pct": 3.5857020979487397, "request_values_pct": [3.690234278881843, 3.4811699170156363], "status": "SMALL_SUBSET_COMPATIBLE", "scope": "separate restart blocks; not randomized interleave; two requests, medians of repeated actual completions"}], "trace_vs_clean": [{"gamma": 4, "median_trace_clean_difference_pct": 70.69300003479695, "status": "DIAGNOSTIC_NONPAIRED_TIME_BLOCK_DRIFT_NOT_CAUSAL_HOOK_OVERHEAD"}, {"gamma": 8, "median_trace_clean_difference_pct": 28.596534961235285, "status": "DIAGNOSTIC_NONPAIRED_TIME_BLOCK_DRIFT_NOT_CAUSAL_HOOK_OVERHEAD"}], "limitation": "do not attribute differences to hooks alone; drift/producer readiness/shared fabric are not experimentally separated"}

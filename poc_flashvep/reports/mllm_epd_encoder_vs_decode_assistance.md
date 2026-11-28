# Matched encoder and decode assistance

The two intervention arms select the same request indices. E routes the selected requests through the extra LM worker colocated with vision encoding and hands off its KV through NIXL. D runs the selected prefills on the decode worker and begins decode from its local KV. The same-runtime control keeps independent P and D and uses the same routed HTTP gateway. The E LM helper entails extra model weight residency. The D gateway holds pending decode requests during the local prefill to first generated token interval. The HTTP intervals include scheduling, stage dispatch and protocol overhead. Native NIXL counters identify connector traffic with the reported operation counts. Three arms execute sequentially. Repeated independent starts and task-quality comparisons would be required for an efficacy claim.

## N4 paired arms

Observed matched helper slots 1/1. 4/4 greedy sequences agree exactly E versus D. Evidence `MEASURED_ONLINE`.

| Arm | Completed | Assisted | Assisted LM input tokens | E HTTP p95 ms | TTFT p95 ms | TPOT p95 ms | ITL p99 ms | Decode SLO | Joint goodput req/s | Non-assisted Decode SLO | NIXL GiB (ops) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `matched-assist/probe_e_smoke` | 4/4 | 1 | 30,326 | 1,470.9 | 3,614.0 | 4.46 | 21.55 | 4/4 | 0.738 | 3/3 | 5.58 (4) |
| `matched-assist/probe_d_smoke` | 4/4 | 1 | 30,326 | 1,256.0 | 2,853.0 | 6.44 | 19.33 | 4/4 | 0.737 | 3/3 | 3.96 (3) |

E-helper P-stage HTTP median 2,178.8 ms. D-local submit to first output median 1,559.0 ms. D proxy reported HTTP-empty admission for 1/1 assisted requests.

## N24 paired arms

Observed matched helper slots 6/6. 23/24 greedy sequences agree exactly E versus D. Evidence `MEASURED_ONLINE`.

For the same-runtime no-assist arm, exact agreement is 23/24 for E assist and 22/24 for D-local.

| Arm | Completed | Assisted | Assisted LM input tokens | E HTTP p95 ms | TTFT p95 ms | TPOT p95 ms | ITL p99 ms | Decode SLO | Joint goodput req/s | Non-assisted Decode SLO | NIXL GiB (ops) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `matched-assist/native_matched_ondemand_n24` | 24/24 | 0 | 0 | 5,848.5 | 11,090.3 | 4.09 | 6.33 | 24/24 | 0.278 | 24/24 | 33.55 (24) |
| `matched-assist/assist_e_n24` | 24/24 | 6 | 155,971 | 11,053.6 | 12,514.7 | 4.05 | 16.32 | 24/24 | 0.129 | 18/18 | 33.55 (24) |
| `matched-assist/assist_d_n24` | 24/24 | 6 | 155,971 | 5,446.3 | 7,898.4 | 5.67 | 16.85 | 24/24 | 0.711 | 18/18 | 25.20 (18) |

E-helper P-stage HTTP median 1,588.9 ms. D-local submit to first output median 1,195.7 ms. D proxy reported HTTP-empty admission for 6/6 assisted requests.

| Relative to same-runtime control | TTFT p95 | TPOT p95 | ITL p99 | Joint goodput | Strict +5% decode tail protection |
| --- | --- | --- | --- | --- | --- |
| E helper | 12.84% | -0.91% | 157.72% | -53.59% | Fail |
| D-local | -28.78% | 38.67% | 166.12% | 155.89% | Fail |

[Matched outcomes figure](../mllm_epd_prefill_heavy_modality_discovery/results/region/figures/encoder_vs_decode_matched_assistance.svg)

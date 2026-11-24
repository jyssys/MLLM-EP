# MLLM100 EPD Motivation Pilot

Evidence: DIAGNOSTIC_C3. Status: preliminary. Task quality remains pending.
Each point uses one server start. These points do not provide a start-level confidence interval.
The initial Natural EOS pilots use Main100. They do not use the later fixed-output runtime controls.
Each method uses three H100 GPUs. All points use the same manifest, request order and arrival seed.
Encoder work runs online. The EC handoff uses the installed debug connector and tmpfs files.
This EC transport is not a production NIXL embedding connector. The P-to-D KV handoff uses NIXL.
The installed source and its local changes must be distinguished from an upstream version label.

## Latency and throughput

| Method | QPS | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Output tok/s | Total tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Agg DP3 | 0.5 | 938.997 | 2291.080 | 8.705 | 7.302 | 27.238 | 5752.825 |
| Agg DP3 | 2 | 1101.237 | 3005.153 | 323.985 | 55.747 | 106.578 | 21739.204 |
| E + PD DP2 | 0.5 | 1509.858 | 3383.556 | 12.007 | 9.484 | 28.433 | 5739.647 |
| E + PD DP2 | 2 | 3938.880 | 6982.663 | 210.096 | 101.068 | 101.558 | 20357.401 |
| Native EPD | 0.5 | 2337.355 | 4839.835 | 16.421 | 7.665 | 26.961 | 5716.013 |
| Native EPD | 2 | 4592.063 | 9372.213 | 24.952 | 8.250 | 93.445 | 19426.833 |

## SLO

Thresholds: TTFT ≤5000ms, every client ITL ≤34ms, request TPOT ≤33ms/token.
Joint PASS requires all three conditions. Every attainment denominator is 100 requests.
A low token-level ITL p99 does not prove that every ITL passes for each request.

| Method | QPS | TTFT PASS % | All-ITL PASS % | TPOT PASS % | Joint PASS % | SLO output tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Agg DP3 | 0.5 | 100.00 | 95.00 | 100.00 | 95.00 | 25.976 |
| Agg DP3 | 2 | 100.00 | 63.00 | 90.00 | 63.00 | 47.072 |
| E + PD DP2 | 0.5 | 100.00 | 87.00 | 100.00 | 87.00 | 22.587 |
| E + PD DP2 | 2 | 63.00 | 46.00 | 85.00 | 38.00 | 26.170 |
| Native EPD | 0.5 | 96.00 | 95.00 | 100.00 | 92.00 | 23.572 |
| Native EPD | 2 | 53.00 | 74.00 | 100.00 | 42.00 | 31.674 |

## Interpretation and limits

At QPS2, Native EPD has lower ITL p99 than Agg DP3. It also has higher TTFT p95.
This is an observed trade-off in an unqualified pilot. It is not a confirmed motivation result.
Native EPD passes the all-ITL gate for 74/100 requests at QPS2. It fails the 95% absolute decode target.
No method or request is removed because its result is unfavorable.
Natural output counts differ between methods. This table does not control output work.
Throughput uses planned first arrival through drain. It is finite-cohort throughput, not sustainable capacity.
Total throughput counts each request input and output once. It does not count tokens again at each E/P/D stage.
TTFT includes dispatch lateness and all stage waits. Client ITL includes delivery delay.
TPOT is (last token time − first token time)/(output count − 1). Single-output TPOT is zero.
Mean TBT averages all client inter-token intervals. Co-delivered tokens have zero intervals.
Runtime host enclosures are not GPU-exclusive service times.
E/P/D physical idle results remain diagnostic. No additional prefill work has established safe headroom.
The main CSV records raw hashes, config hashes, counts and durations.
CSV: `mllm_epd_prefill_heavy_token_split_20261124/analysis/main100_baseline_motivation_pilot.csv`.
Figures: `mllm_epd_prefill_heavy_token_split_20261124/figures/main100_motivation_*.{png,pdf}`.
Each figure has a matching CSV. No omitted QPS is interpolated or represented as a measured point.

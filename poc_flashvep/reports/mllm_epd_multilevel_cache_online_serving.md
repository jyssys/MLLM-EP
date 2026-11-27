# Online evaluation of passive E-side caching

## Natural-content reuse

Natural149 keeps original image and question pairs and frozen manifest order. The local materialized media are enriched in long documents. Arrival is a 2.5 req/s seeded Poisson trace. Both arms use original BF16 checkpoints, 1E/1P/1D, free decode batching, NIXL and Natural EOS. The first measured sequence starts after a different calibration request. The second replays the complete measured sequence. Each row belongs to a single independent server start per method.

| Method | Cohort | Completed | TTFT p95 s | TPOT p95 ms | ITL p99 ms | E2E p95 s | Joint SLO | Goodput req/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NativeStrong | first | 149 | 17.656 | 5.043 | 12.187 | 18.517 | 67/149 | 0.898 |
| Scoped E embeddings + native P/D APC | first | 149 | 11.653 | 7.465 | 17.480 | 12.018 | 70/149 | 1.202 |
| NativeStrong | repeated | 149 | 17.901 | 5.069 | 11.849 | 18.718 | 67/149 | 0.894 |
| Scoped E embeddings + native P/D APC | repeated | 149 | 2.839 | 5.693 | 14.301 | 3.171 | 147/149 | 2.523 |

The first cache-enabled run reduced TTFT p95 by 34.0% and raised finite-cohort joint goodput by 33.9%. TPOT p95 increased 48.0% and ITL p99 increased 43.4%. The repeated cohort reduced TTFT p95 by 84.1% and still increased TPOT p95 by 12.3% and ITL p99 by 20.7%. These pairs therefore fail the user-defined 5% decode nonregression gate. The separate starts and different E-side renderer and transfer path bound attribution. NativeStrong kept its original inline transport. The cache arm uses a single E renderer and source leases. Both arms collect the same one-second Prometheus samples.

First sequence encoder observations were 1,509 actual resident image hits and 1,130 encoder computations among 2,639 occurrences. In the repeated sequence all 2,639 acquisitions hit and the registry reported zero encoder computations. About 6.83 GiB of unique idle source storage remained after drain. The P HTTP API reported 896,000 and 1,031,168 cached prompt rows in the two sequences. The stock engine summed 982,144 and 1,133,568 prefix-hit counter rows respectively, which can include scheduler queries beyond the request-final API usage. E HTTP p95 was 10.846→3.353 s on the first sequence and 10.405→1.666 s on the repeat. P HTTP p95 increased 8.083→8.873 s on the first sequence as E processed requests faster. Each interval includes host-side work and queueing.

## Numerical fidelity and quality

Native A/A within the same server start matched 137/149 complete Natural outputs and 149/149 first tokens. Cache+APC versus Native matched 124/149 full outputs and 147/149 first tokens on the first sequence, and 128/149 full outputs and 146/149 first tokens on the repeated sequence. The trace changes compute and batch geometry. These results allow drift quantification and cannot establish all-layer correctness by themselves.

On the first sequence the pinned local scorer gave identical scores in both methods: Mantis accuracy 48%, DocVQA raw ANLS 0%, MP-DocVQA raw ANLS 5.67%, and MMLongBench-Doc raw incomplete score 2.5%. No paired score changed under these scorers. DocVQA floor and the unavailable official MMLongBench-Doc extraction limit the task quality conclusion. Task quality noninferiority remains unqualified.

[Online raw hashes and reduced record](../mllm_epd_memory_asymmetry_discovery/results/online_pilot.json) · [Natural149 task scores](../mllm_epd_memory_asymmetry_discovery/results/quality_natural149_pilot.json) · [Latency and goodput](../mllm_epd_memory_asymmetry_discovery/figures/natural_online.png) · [Matched request timeline](../mllm_epd_memory_asymmetry_discovery/figures/native_cache_http_timeline.png)

## Controlled working-set comparison

The 188-request trace cycles through a 54.42 GiB first wave and then revisits the same document order. The second wave combines 35 original follow-ups and 59 explicitly labeled exact prompt repeats. All methods use fixed16, the same bounded E embedding cache, 42 GiB D KV and the same stable handoff. Every N1 arm completed 188/188.

| P and remote KV GiB | P requests with cached rows | TTFT p95 s | TPOT p95 ms | ITL p99 ms | Joint SLO | Goodput req/s |
|---|---:|---:|---:|---:|---:|---:|
| P42 local APC | 1 | 19.635 | 4.814 | 8.867 | 49/188 | 0.577 |
| P72 local APC | 94 | 18.864 | 11.009 | 47.593 | 93/188 | 1.244 |
| P8 + host64 | 94 | 20.559 | 20.124 | 48.154 | 86/188 | 1.153 |
| P8 + E64 peer | 94 | 20.746 | 8.367 | 35.987 | 88/188 | 1.179 |
| P16 + E56 peer | 94 | 19.163 | 30.282 | 76.584 | 94/188 | 1.256 |
| P42 + E64 peer | 94 | 19.563 | 17.327 | 67.333 | 84/188 | 1.122 |

The P42 local arm reused 27,008 prefix rows. Each larger cache placement reused 1,018,624 rows. The peer and pinned-host arms loaded 54.400 GiB of physical KV at the same reuse count. For P8, GPU-event load+store enclosures summed to 0.675 s peer and 2.330 s host. The sums include overlapping work and differ from critical-path wall time. The P8 + E64 peer tier physically allocated 63.998 GiB on E with a 32 GiB startup guard. E reported 40.478 GiB free at drain. Full fixed-length token output matched 183/188 vs host and 179/188 vs local P72. The first-token agreements were 188/188 and 187/188.

The P8 variant reached a sampled 97.6% peak in active P KV. Increasing active P room to 16 or 42 GiB reduced its occupancy peak to 73.5% or 44.3% and reduced TTFT p95 compared with P8. Against the equal-total-budget P72, P16 + E56 gained 1.0% goodput and incurred 1.6% higher TTFT p95, 175.1% higher TPOT p95 and 60.9% higher ITL p99. Giving the peer path a full P42 allocation provided no gain over local P72. P42 + E64 preserved P42's active KV budget and about doubled its joint SLO goodput, with TPOT and ITL far beyond the allowed relative guard. The D decoder retains active KV locally in every arm. Admitted traffic and scheduler/transfer interference can still change decode tails. No engine timing data isolate a single source for the difference.

[Working-set result, timing and transfer counters](../mllm_epd_memory_asymmetry_discovery/results/online_working_set_pilot.json) · [Trace construction](../mllm_epd_memory_asymmetry_discovery/workload_audit.md)

## Representation ablations at 2.5 req/s

| First sequence, same stable bridge | E image hits | P requests with cached rows | TTFT p95 s | TPOT p95 ms | ITL p99 ms | Goodput req/s |
|---|---:|---:|---:|---:|---:|---:|
| No E retention or APC | 0 | 0 | 43.656 | 5.060 | 11.144 | 0.518 |
| E embeddings only | 1,509 | 0 | 23.757 | 5.120 | 10.969 | 0.572 |
| P/D APC only | 0 | 17 | 42.391 | 5.203 | 11.191 | 0.493 |
| E embeddings + P/D APC | 1,509 | 54 | 11.653 | 7.465 | 17.480 | 1.202 |

The same stable bridge processed 2,639 original image occurrences. E-only reduced its E HTTP p95 to 2.582 s, then P HTTP reached 22.978 s. APC-only reused 221,952 rows, versus 896,000 with both caches. The actual P-submission trace had 55 ordered-image revisits in each arm. The preceding same-document P HTTP request was still unfinished at submit for 52/55 APC-only revisits and 38/55 combined revisits. These host enclosures show a changed request pipeline and do not identify the exact prefix-block availability time. The single-cache arms improve TTFT relative to the matched bridge with both caches disabled. Both lag optimized inline NativeStrong in TTFT and goodput at this offered load. The ablations demonstrate why stage queueing and overlap matter alongside nominal cache capacity. The combined arm still fails decode nonregression against the optimized inline NativeStrong baseline.

## Matched embedding storage

This comparison keeps the APC, scope, cache budget and E→P CUDA IPC contract fixed. The host arm keeps an immutable pinned representation between requests and restores a temporary E buffer. A P-local CPU design that skips E HTTP could have different costs. In the first pair GPU retention yielded 11.653 s TTFT p95 and 1.202 req/s goodput, versus 23.639 s and 0.610 req/s on host. The host path copied 6.802 GiB into pinned storage during the first measured sequence and spent 13.306 seconds in the summed host store enclosures. Direct attribution of the entire TTFT gap requires a stage intervention. In the second sequence, when neither arm stored new entries, host and GPU yielded 2.768 and 2.839 s TTFT p95, and 2.562 and 2.523 req/s goodput. The host path rehydrated 2,447 embeddings, moved about 12.073 GiB, and spent 0.471 seconds in summed load enclosures. The physical E-VRAM advantage appears chiefly during fill in this pair.

## Zero-reuse control

Unique91 preserves 91 original requests with no cross-request image reuse. At 2.5 req/s Natural EOS, optimized NativeStrong completed 91/91 with 10.675 s TTFT p95 and 1.387 req/s goodput. The stable bridge with embeddings + local APC completed 91/91 with 19.258 s and 0.794 req/s. Adding the 16 GiB peer tier to that same bridge produced 17.192 s and 0.948 req/s. Its E image and P prefix hit counts were zero. It inserted 52.643 GiB on E and never loaded prefix pages. The comparison rules out deploying this whole bridge unconditionally on the zero-reuse trace. Within the stable-bridge comparison, an N1 timing difference cannot isolate a write penalty. Positive insertion bytes with zero loads directly establish wasted movement under this workload.

## Offered-load sensitivity

The first Natural149 sequence compares NativeStrong and the scoped E-embedding plus P/D APC stack. All arms completed 149/149 at seed 2026112402. QPS is the offered rate. The completion and goodput rates include trace drain and do not estimate sustainable capacity.

| Offered req/s | TTFT p95 Native / cached s | Joint goodput Native / cached req/s | TPOT p95 Native / cached ms | ITL p99 Native / cached ms |
|---:|---:|---:|---:|---:|
| 1.0 | 2.820 / 2.780 | 1.028 / 1.040 | 5.066 / 5.104 | 10.872 / 11.197 |
| 2.5 | 17.656 / 11.653 | 0.898 / 1.202 | 5.043 / 7.465 | 12.187 / 17.480 |
| 5.0 | 34.066 / 24.835 | 0.796 / 0.853 | 5.073 / 7.345 | 12.920 / 23.584 |

Low load has little primary-metric headroom. The higher load points reduce TTFT as E/P reuse increases aggregate service, and both exceed the 5% decode protection margins. At 5 req/s the number of joint-SLO requests fell from 51 to 42 despite faster drain. [Load sweep data](../mllm_epd_memory_asymmetry_discovery/results/online_load_sweep.json)

### Three independent placement pairs

The confirmation independently restarted both methods three times at 2.5 req/s and alternated which method ran first. Both had 12 GiB of retained-embedding budget, 42 GiB local P KV, 42 GiB local D KV, identical calibrated arrivals and a complete warm replay. The optimized pinned-host implementation fused its D2H submissions into 94 batches per first sequence and fenced once per batch. Actual store enclosures totaled 6.089–7.588 seconds over the three first sequences. No pinned copy remained pending at drain. All 1,788 timed requests completed. This comparison tests E VRAM and pinned host placement using the same E-side bridge.

| Metric | Cold VRAM / host mean ratio [paired 95% CI] | Warm VRAM / host mean ratio [paired 95% CI] |
|---|---:|---:|
| TTFT p95 | 0.811 [0.760, 0.855] | 1.230 [0.993, 1.597] |
| E2E p95 | 0.806 [0.763, 0.841] | 1.199 [0.973, 1.548] |
| Joint goodput | 1.295 [1.084, 1.677] | 1.012 [1.008, 1.020] |
| TPOT p95 | 1.035 [0.939, 1.103] | 0.992 [0.979, 1.010] |
| ITL p99 | 0.915 [0.833, 1.050] | 0.992 [0.950, 1.018] |

VRAM lowered cold TTFT in each independent pair. The upper cold TPOT interval still exceeds the 5% guard, and the repeated sequence has no material goodput improvement. These CIs resample three independent paired starts and are coarse. Local task-score point estimates matched in all six phase/placement pairs. The DocVQA score floor and missing official MMLongBench-Doc extraction remain. Cold fill gives a scoped E-VRAM placement effect. This does not qualify a general cache placement rule or a joint online win. [Cold ratios](../mllm_epd_memory_asymmetry_discovery/results/paired_embedding_placement_cold.json) · [Warm ratios](../mllm_epd_memory_asymmetry_discovery/results/paired_embedding_placement_warm.json) · [Paired figure](../mllm_epd_memory_asymmetry_discovery/figures/embedding_placement_paired.png) · [Placement quality](../mllm_epd_memory_asymmetry_discovery/results/quality_embedding_placement.json)

[All EPD serving panels](../mllm_epd_memory_asymmetry_discovery/results/online_serving_panels.json) · [Cold and warm at the studied load](../mllm_epd_memory_asymmetry_discovery/figures/natural_online.png) · [Offered-load figure](../mllm_epd_memory_asymmetry_discovery/figures/natural149_offered_load.png) · [Working-set placements](../mllm_epd_memory_asymmetry_discovery/figures/working_set_online.png) · [Zero-reuse figure](../mllm_epd_memory_asymmetry_discovery/figures/unique_online.png)

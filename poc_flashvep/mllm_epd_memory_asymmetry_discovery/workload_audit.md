# Reuse and cache-capacity workloads

This audit reuses the original public questions, order of images within each request, compiled model inputs, processor geometry and BF16 precision. Main400 is read from the original Git object at `07892cbe7193366ec422f1df82006c6e69f91cec` and verified with SHA-256 `7ac7d28ae1678499bee35ec20e317760555944cac7b4ca29eb46aeb3957b9592`. Its 400 processor metadata files are available. The historical asset checkout contains all images for 149 of the requests, including Main100. Available images and metadata were verified against the original digests. The resulting media-available subset contains 80 MMLongBench-Doc, 31 MP-DocVQA, 25 Mantis-Eval and 13 DocVQA requests. Media availability increases the share of long-document requests in this subset.

## Prior-order causal reuse

A reusable image occurrence requires the same normalized pixel digest, grid, processed resolution, model revision and processor revision in an earlier request. The exact-prefix analysis hashes the ordered prompt token IDs, all three MRoPE axes, multimodal token types, and content identity of every visual slot, with the original model, template and processor version and a public-study scope. Hash chains preserve causal history. The analysis reserves at least the final row for the logits path and reports whole 128-row block matches. These are causal opportunities before capacity, cache state, queueing or transfer effects. A stable image sequence can admit exact Prefix KV reuse where the question follows it. Shared images in different sequences can still reuse their embedding.

| Original order | Main100 | Main400 | Natural149 | Unique image control |
|---|---:|---:|---:|---:|
| Requests | 100 | 400 | 149 | 91 |
| Image occurrences previously seen | 123/1,251 | 2,263/5,004 | 1,511/2,639 | 0 |
| Requests with a prior image occurrence | 8 | 94 | 57 | 0 |
| Same ordered image sequence, distinct question | 6 | 89 | 55 | 0 |
| Requests with exact 128-row prefix match | 6 | 90 | 55 | 0 |
| Exact prefix rows / prompt rows | 77,056/1,102,959 | 1,412,224/4,408,620 | 913,664/1,943,636 | 0 |
| Scoped document-ID coverage | 75/100 | 300/400 | 124/149 | 68/91 |
| Requests with a repeated scoped document ID | 7 | 91 | 56 | 1 |

Main400's median prior reuse distance is 32 arrivals for images and 29 for complete ordered image sequences. Natural149's medians are 18 and 15 respectively. A controlled grouping of the same Natural149 inputs makes both medians one arrival. The grouped workload is a locality-positive control. The original order comes from the frozen experiment manifest. It preserves natural question-to-document relations and has no production arrival timestamps.

| Source | Main400 prior image occurrences | Natural149 prior image occurrences | Natural149 ordered-image follow-ups |
|---|---:|---:|---:|
| DocVQA | 2/50 | 1/13 | 1/13 |
| Mantis-Eval | 7/254 | 2/64 | 0/25 |
| MP-DocVQA | 39/312 | 20/98 | 6/31 |
| MMLongBench-Doc | 2,215/4,388 | 1,488/2,464 | 48/80 |

Within-source image occurrence denominators count images. The follow-up denominators count requests. Most reusable rows come from MMLongBench-Doc in both cohorts.

Natural149 has 35 repeated document-shaped ordered image groups, with 90 request members. In Main100, the original Mantis examples `wc_22` and `wc_30` share one of two images, and `xh_18` and `xh_7` share one of three. Both pairs have an embedding reuse opportunity with zero whole-block Prefix KV match. These data support partial-document reuse and standalone follow-up questions. A growing conversation-history replay and a natural cross-tenant trace remain `NOT_RUN`. All requests use a public benchmark isolation scope.

For Natural149 the p50/p90 payload per request is 2.234/7.554 MiB of stored compressed image derivatives, 86.133/168.438 MiB of LM-ready BF16 embeddings, and 692.781/1,354.073 MiB of logical Full28 prompt KV. Embedding storage uses 7,168 bytes per visual row. Qwen2.5-VL uses 57,344 bytes per prompt KV row before allocator padding. Physical timing and safety are measured independently.

[Raw reuse audit and reuse-distance distributions](results/workload_reuse.json) · [Cohort recurrence figure](figures/workload_reuse_distribution.png)

## Whole-block LRU replay

The sequential replay processes each frozen request to completion before the next arrival. Its separate LRU embedding and KV partitions have the same total payload budget. KV hits preserve the final prompt row for logits. Crediting an encode bypass from a KV hit requires an implementation able to suppress E dispatch and a full covered image span. The current replay has no live pages, concurrent copying, latency or stock APC free-list model.

On Natural149 in prior order, a 4 GiB embedding-only cache retains all 1,511 prior-image hits even though its distinct occupied working set over the trace is 6.78 GiB. A 4 GiB KV-only cache yields just 7 exact block hit requests, skipping 4.47% of prompt rows. At 16 GiB, KV-only reaches 38 requests and 31.52% of prompt rows. A 16 GiB 50:50 partition yields 21 Prefix KV hit requests and keeps the full 47.37% image row avoidance. Increasing the KV portion of the equal-budget hybrid can give more prefix retention. The grouped order reaches the same maximal embedding hits with about 0.25 GiB and its full prefix hits at about 2 GiB. A favorable order changes the capacity regime materially.

[Whole-block replay](results/cache_replay.json) · [Capacity curves](figures/workload_cache_capacity_replay.png)

## Exact-prefix eviction controls

The natural two-wave149 trace starts with 94 first requests, then uses their 55 actual distinct follow-up questions. The controlled stress188 trace uses the same first 94 requests then exactly one request for each group, in the same cyclic order. The stress second wave has 35 distinct real follow-ups and 59 full request replays. Every replay is explicitly tagged, and its question, media, token metadata and processor contract stay unchanged. The first wave totals 54.420898 GiB of unique whole-block prompt KV. With a maximum input of 30,536 rows, one request uses about 1.63 GiB of KV. The scoped Poisson traces use the original client's exponential interarrival formula, seed 2026112402, and offered rates 1, 2.5 and 5 request/s. The QPS2.5 second wave starts at 36.8877 seconds. The external manifest and arrival receipts preserve order and content hashes.

| Offline model | Second-wave hits, natural149 | Second-wave hits, stress188 |
|---|---:|---:|
| 42 GiB local, no tier | 51/55 | 0/94 |
| 8 GiB local, no tier | 20/55 | 0/94 |
| 8 GiB local + 34 GiB peer, equal total bytes to local 42 GiB | 34/55 | 0/94 |
| 8 GiB local + 48 GiB peer | 54/55 | 0/94 |
| 8 GiB local + 64 GiB peer | 55/55 | 94/94 |
| 42 GiB local + 64 GiB peer | 55/55 | 94/94 |

The byte-bounded inclusive tier keeps a block unreadable until it is durable. In stress188, first-touch admission with 64 GiB peer yields 1,018,624 remotely reused rows, about 54.4 GiB read. Four completed requests of modeled writeback delay and 2 GiB of local active reserve still preserve all 94 stress hits due to the long time between waves in request order. With two observations required before store, zero stress replay requests hit the peer. Natural149 gains just four remote hits from a 64 GiB tier alongside stock 42 GiB local, after 54.42 GiB of cold insertion. These quantities assume ideal cache lookup and separate pending staging. Actual vLLM admission derives frequency from connector lookup history, which can stop at the first miss. Accumulated payload and occupancy do not imply runtime service-time savings.

[Stress manifest and arrival receipts](results/capacity_stress_workloads.json) · [Durable tier replay](results/passive_tier_screen.json) · [Eviction control figure](figures/workload_cache_stress.png)

## Matched representation policies

The atomic checkpoint replay compares embedding-only, prefix-only, three static partitions and two unified policies at identical combined bytes. An embedding entry stores one complete LM-ready visual tensor. A prefix entry stores an exact checkpoint up to a whole 128-row block near the last image closing marker. Unified policy B7 scores observed frequency times estimated net benefit per byte. B8 adds only a bounded multiplier for an embedding observed in multiple exact causal prefix scopes. Each policy sees past requests and live contents only. Identical connector geometry is used for both. A second replay interpolates GPU event calibration from the independently measured isolated sample and keeps stage demand separate from KV insert/read costs. These models give `OFFLINE_ORACLE` evidence and omit Decode, queues, interference, runtime tracker cost, and actual serving latency.

With Prefix-to-Encode bypass disabled, the test approximates a connector path that already dispatched E before the P cache decision. At 4 GiB on Natural149, embedding-only delivered 1,511 image hits while the calibrated unified policy delivered 760 image hits and three prefix hits. The embedding-only policy left a modeled 64.76 E and 72.81 P seconds of aggregate compute demand. The unified policy left 91.41 E and 71.58 P seconds. At 16 GiB, static 25:75 yielded the same 1,511 image hits and 29 prefix hits, versus 22 prefix hits for either unified policy. At 64 GiB, static 25:75 and both unified policies all yielded 1,511 image hits and 55 prefix hits. The scope feature preserved both hit counts in these Natural149 comparisons. On stress188, its 4 GiB benefit was two extra image occurrences and zero extra prefix hits. Unique91 produced zero hits under all policies. The observed presence of two valid representation types identifies a trade-off, with no demonstrated local scope-feature benefit over frequency/cost history.

[Geometry-only matched replay](results/matched_policy_screen_geometry.json) · [Calibrated matched replay](results/matched_policy_screen_calibrated.json) · [Matched policy figure](figures/workload_matched_policy_screen.png) · [Physical calibration](results/physical_economics_summary.json)

## Future-aware B9 reference

B9 uses the same exact object geometry, byte budget, and copy accounting as B4–B8. It ranks each object by calibrated benefit per byte divided by the distance to its next known compatible request, and admits it when enough lower-scored bytes can be evicted. A read requires a completed earlier write. The future order information disqualifies this policy from online use. The bounded replay is a hindsight heuristic with no optimality guarantee. A separate unlimited-capacity replay reports per-stage coverage bounds in the same exact checkpoint geometry.

| Atomic checkpoint replay, no prefix Encode bypass | 4 GiB | 16 GiB | 64 GiB |
|---|---:|---:|---:|
| Natural149 B9 prefix hit requests | 23 | 50 | 55 |
| Natural149 causal B7 prefix hit requests | 3 | 22 | 55 |
| Stress188 B9 prefix hit requests | 0 | 26 | 94 |
| Stress188 causal B7 prefix hit requests | 0 | 0 | 94 |

Natural149 B9 reaches all 1,511 embedding hits at these budgets. At 4 GiB its modeled maximum E/P compute demand is 64.76 s versus 72.68 s for the best tested causal policy. At 16 GiB, B9's extra P hits leave that proxy unchanged relative to static 25:75 because the calibrated E stage already dominates. The bound permits separate comparison by stage and gives no TTFT or goodput estimate. All policies have zero positive hits on Unique91.

The first Natural149 QPS2.5 combined cache pilot measured 1,509 image acquisitions flagged as resident, 54 positive P APC requests, and 896,000 cached prompt rows. The distinct full-block prior-order reference has 1,511 image occurrences, 55 positive requests, and 913,664 cached prompt rows. Realized E and P row coverage is 99.9% and 98.1% of those respective references. This cross-check uses the stock whole-block APC contract and differs from B9's atomic checkpoint geometry. The online requests also overlap, while the prior-order reference gives each previous request time to complete.

[B9, relaxed coverage, and Natural149 counter cross-check](results/b9_future_aware_oracle.json) · [Measured online pilot](results/online_pilot.json)

## Reproducibility

From the research worktree, use the pinned runtime interpreter and historic asset checkout. All generated manifest and timing paths below stay outside Git. Compact JSON results and figures remain in the experiment namespace.

```bash
pkg=poc_flashvep.mllm_epd_memory_asymmetry_discovery
python=../mllm_epd_runtime_env/bin/python
run_root=../h200_memory_asymmetry_runs
asset_repo=/home1/irteam/MLLM_PD/MLLM-EP-mllm-epd.filtered
main100=/home1/irteam/MLLM_PD/mllm_epd_autosetup/manifests_1795600931728841878/main100_manifest_template_v2.json
result=poc_flashvep/mllm_epd_memory_asymmetry_discovery/results

"$python" -m "$pkg.workload" --asset-repo "$asset_repo" --main100 "$main100" \
  --run-root "$run_root" --results "$result"
"$python" -m "$pkg.cache_replay" --asset-repo "$asset_repo" --main400 \
  --manifest "$run_root/natural149_manifest.json" --manifest "$run_root/natural149_grouped_manifest.json" \
  --manifest "$run_root/unique_images_manifest.json" --out "$result/cache_replay.json"
"$python" -m "$pkg.stress_workload" --asset-repo "$asset_repo" --run-root "$run_root" \
  --natural149 "$run_root/natural149_manifest.json" --results "$result"
"$python" -m "$pkg.tier_screen" --asset-repo "$asset_repo" --out "$result/passive_tier_screen.json" \
  --manifest "$run_root/w5_eviction188_manifest.json" --manifest "$run_root/w5_natural_two_wave149_manifest.json"
"$python" -m "$pkg.policy_screen" --asset-repo "$asset_repo" --out "$result/matched_policy_screen_geometry.json" \
  --manifest "$run_root/natural149_manifest.json" --manifest "$run_root/unique_images_manifest.json" \
  --manifest "$run_root/w5_natural_two_wave149_manifest.json" --manifest "$run_root/w5_eviction188_manifest.json"
"$python" -m "$pkg.policy_screen" --asset-repo "$asset_repo" --out "$result/matched_policy_screen_calibrated.json" \
  --economics "$result/physical_economics_summary.json" \
  --manifest "$run_root/natural149_manifest.json" --manifest "$run_root/unique_images_manifest.json" \
  --manifest "$run_root/w5_natural_two_wave149_manifest.json" --manifest "$run_root/w5_eviction188_manifest.json"
"$python" -m "$pkg.b9_oracle" --asset-repo "$asset_repo" --out "$result/b9_future_aware_oracle.json" \
  --economics "$result/physical_economics_summary.json" \
  --manifest "$run_root/natural149_manifest.json" --manifest "$run_root/unique_images_manifest.json" \
  --manifest "$run_root/w5_eviction188_manifest.json" \
  --online-pilot "$result/online_pilot.json" --workload-receipt "$result/workload_reuse.json"
```

Use a Python environment with matplotlib for `python -m "$pkg.workload_figures" --results "$result" --out "$(dirname "$result")/figures"`.

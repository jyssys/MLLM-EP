# Resume checkpoint — 2026-09-09 12:27 KST

GPU work resumed at 10:27 after user-cleared physical GPUs 4–7. Two hours since
resume is not two hours of CUDA busy time: model loading, analysis, warmup and
per-device experiment interval accounting are separate. The prior user-requested
overnight GPU pause is not active experiment time.

## Strongest positive (original baseline, not successor)

Layered cap4 versus chunk512 improves bursty mean request E2E by median 19.67%
across three graph-enabled engine pairs. Native route/layer instrumentation
confirms prefill is concentrated in active groups, reducing repeated prefill
expert-use occurrences. This is the original mechanism, with a known TTFT/TBT
tradeoff. Best-static cap4 wins both current workload medians; no additional
per-regime E2E envelope among cap4/cap16.

## Strongest negative

FastPP's three-policy native MoE envelope adds only 0.59–2.35% mean request E2E.
The first fresh static-partition pair reverses a 23–26% stage-max proxy: actual
request E2E worsens 11.74% bursty / 14.91% steady. Only one complete pair so far;
do not promote this sign without the other two. A rank-local profile is not
automatically portable to a differently owned PP stage or a request timeline.

## Open issues / fidelity

- FastPP restart variation: full-workload warmup and fixed conservative KV
  capacity controls are running. Uneven PP's native KV-count mismatch was fixed
  only in the isolated reference with a red/green CPU regression; short native
  answer sanity passes. It is a compatibility fix, not a proposed method.
- NanoFlow overlap is real but the profiled B16 split2 is slower. Independent HF
  numerical mismatches concentrate at near ties. Full task-quality equivalence
  remains unclaimed. Pure-prefill adapter first-use failed because unplanned
  attention algorithms were uninitialized; stopped and CPU-tested correction
  prepared, with fresh smoke pending. No result from failed runs is retained as
  performance. No paper searched-optimum claim for the manual plan set.
- Qwen3-VL clean/lite control: 1,248 measured requests across six engines,
  14,399 measured logical MoE samples with exact four-rank joins, no unknown request
  IDs. Observer overhead median 2.54–6.54% across families. Keep clean request
  E2E separate. The zero-TTFT fixed-timeline prefill-only bound is 6.08–11.52%,
  not a total scheduling or online queue bound.

## Next experiments

1. Complete FastPP static-partition pairs; compare against equal partition with
   the same full workload warmup. No claimed benefit from a layer-only proxy.
2. Retest NanoFlow pure-prefill initialization, then bounded native FFN split /
   existing green-context resource control at M4096/M8192 and decode cohorts.
3. Additional official Layered caps with matched full-workload warmup.
4. Extend real-image transfer to four original ChartQA images, 1.7K–1.9K and
   3.8K–4.0K prompt tokens, exactly/nearly volume-matched natural-text controls.
   No synthetic routing, upsampled images, or old timing reuse. CPU processor
   verifies <=1% paired token-count differences and no CUDA initialization.

No finalist, Kimi run, new optimizer or successor prototype is justified yet.
Prior-art attacks already flag generic dynamic grouping, nano-plan selection and
phase-resource overlap as crowded. No burn co-runs with measurements.

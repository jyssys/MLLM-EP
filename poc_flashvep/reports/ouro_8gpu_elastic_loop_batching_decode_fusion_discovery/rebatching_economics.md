# KEEP versus REFORM component interventions

Eight deterministic state recreations compare KEEP4, YIELD1, and fixed-membership D1.
They use P256/P2048 with one or two old rows and one or two newly eligible rows.
Every hidden/KV state comes from actual pretrained R4 execution and a valid decode prelude.
All three arms reproduce the same initial full-hidden and sampled-KV fingerprint.
They generate the same token IDs in every observed comparison.

The lane uses native synchronous execution.
It is not an optimized-async serving result.
New rows are already prefetched through their prelude.
Their recurrent eligibility starts at the old cohort's R1 device boundary.
That event is not a fresh API arrival or a completed fresh prefill.

Every possible row width receives graph warm-up first.
No v2 intervention captures a new graph inside its measured interval.
The confounded v1 timings remain preserved but excluded.

| P | Old/new rows | KEEP4 old completion ms | YIELD1 old completion ms | KEEP4 all completion ms | YIELD1 all completion ms | YIELD all-row speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
|256 |1/1 |18.302 |20.926 |37.789 |26.220 |1.441× |
|256 |1/2 |18.587 |20.725 |39.361 |26.410 |1.490× |
|256 |2/1 |19.783 |20.980 |39.338 |26.230 |1.500× |
|256 |2/2 |19.682 |21.384 |40.359 |27.038 |1.493× |
|2048 |1/1 |21.339 |23.951 |43.886 |29.933 |1.466× |
|2048 |1/2 |21.327 |24.307 |46.186 |30.843 |1.497× |
|2048 |2/1 |22.283 |24.418 |44.521 |30.441 |1.463× |
|2048 |2/2 |22.422 |24.688 |46.054 |31.178 |1.477× |

These are LIVE_MEASURED controlled components.
Each state has one paired intervention, not three independent process repetitions.
The table therefore does not claim a workload-population confidence interval.
It also does not report an all-workload oracle or an online policy speedup.

## Two effects coexist

KEEP4 completes the original rows sooner but delays newly eligible rows.
YIELD1 co-batches work and completes the combined set sooner.
This is a real priority/packing trade-off.
It does not establish that an adaptive fusion policy beats native Mixed.

The no-reform D1 arm gives the old rows the same priority as KEEP4.
KEEP4 reduces their component latency by a factor of1.039×–1.081× in these sync cases.
That isolates an exposed sync-boundary cost under fixed membership.
The primary native-async clean lane does not retain that low-N benefit.
An ELB claim must beat the existing async execution path, not only a serialized control.

`paired_interventions.csv` contains all three arms and fingerprints.
The raw timelines distinguish eligibility, selection, output delivery, and combined completion.
P2 still needs a useful incremental operating region beyond native/fair Mixed and tuned fixed D.

# Conventional TP, DP, and hybrid controls

## Result

Both new TP2×DP2 jobs complete on the frozen constant-token workload.
The combined audit now admits 400 clean trials across 80 summary cells and ten jobs.
These are generic controls, not candidate architecture results.
The cohort contains two repeated input patterns, not 400 independent natural requests.

TP4 has the lowest observed median in 14 of 16 four-GPU cells.
The other two cells are C8/N4096.
Their numerical gaps are below 0.4% and do not establish a reliable preference.
The jobs are sequential, and TP/DP timer boundaries differ.
No randomized paired architecture comparison or population confidence interval follows.

The current output is [complete_v3](../analysis/native_hybrid_cohorts_complete_v3/summary.json).
The earlier v1 snapshot has only one completed hybrid size.
The v2 values match v3, but its aggregate-KV sentence is incorrect.
V3 corrects that sentence without changing raw trials or numerical results.

## Timing and work contract

TP1/TP2/TP4 use one native generate API call for cohort wall time.
DP4 and TP2×DP2 use the observed parent release-to-all-completed interval.
The parent interval includes release sends, matching receipts, and controller work.
It is not an inferred maximum of worker times.
All workers report ready before the parent timer starts.
The controller adds no per-token cross-replica barrier.

Each cell runs five clean repeats after three short warmups.
The separate trace lane does not enter these medians.
C denotes a submitted finite cohort, not a constant active batch size.
C1 uses the first pattern.
C2/C4/C8 alternate the same two patterns.

Every request emits 128 copies of frozen token 378.
The sampler ignores EOS only for this hardware control.
The scripts check prompt IDs, output IDs, finish reasons, denominators, and source identities.
This does not evaluate model quality or natural-generation throughput.

Every native rank reserves 52 GiB of KV memory.
TP4, DP4, and TP2×DP2 therefore each reserve 208 GiB across four ranks.
Their replica ownership and weight multiplicity still differ.
TP1/TP2 use fewer ranks and smaller aggregate reservations.
No KV footprint is a measured HBM-traffic counter.

## Observed four-GPU medians

All entries are milliseconds.
Keep the API/parent labels when using this table.

| Base | C | Prompt tokens | TP4 API | TP2×DP2 parent | DP4 parent |
|---|---:|---:|---:|---:|---:|
| 1.4B | 1 | 1024 | 813.429 | 994.266 | 1046.912 |
| 1.4B | 1 | 4096 | 863.745 | 1073.953 | 1214.872 |
| 1.4B | 2 | 1024 | 899.904 | 968.941 | 1046.604 |
| 1.4B | 2 | 4096 | 993.632 | 1121.203 | 1213.399 |
| 1.4B | 4 | 1024 | 945.336 | 1040.467 | 1091.817 |
| 1.4B | 4 | 4096 | 1144.338 | 1208.624 | 1216.290 |
| 1.4B | 8 | 1024 | 1014.667 | 1134.365 | 1206.251 |
| 1.4B | 8 | 4096 | 1450.209 | 1445.441 | 1446.820 |
| 2.6B | 1 | 1024 | 1628.599 | 1871.042 | 2097.563 |
| 2.6B | 1 | 4096 | 1718.983 | 2030.435 | 2398.698 |
| 2.6B | 2 | 1024 | 1797.888 | 1884.316 | 2097.841 |
| 2.6B | 2 | 4096 | 1984.755 | 2019.527 | 2397.719 |
| 2.6B | 4 | 1024 | 1893.501 | 2035.081 | 2120.216 |
| 2.6B | 4 | 4096 | 2261.567 | 2304.985 | 2401.673 |
| 2.6B | 8 | 1024 | 1999.824 | 2166.237 | 2304.716 |
| 2.6B | 8 | 4096 | 2877.707 | 2903.111 | 2868.595 |

The complete CSV includes TP1 and TP2 controls, ranges, and worker intervals.
The parent-minus-longest-worker residual includes more than IPC.
For the new 1.4B hybrid cells, its median ranges from 0.299 to 0.406 milliseconds.
This small charged residual does not explain all differences from an earlier isolated TP2 run.
This report identifies no cause for that cross-job difference.

## Physical ownership and startup qualification

The hybrid has two independent native TP2 groups: physical [4,5] and [6,7].
Each group maps those GPUs to logical ranks [0,1].
The runner checks the exact UUID order and world size 2.
It also checks eight local Q/KV heads and four depth banks per physical layer.
Every declared process retains the required UID, session, process group, ancestry, and guard identity.
C1 leaves one reserved replica idle.
Its two GPUs are not reported as useful C1 compute or as free devices.

The first three 1.4B attempts fail before valid timing.
Their raw failures remain intact.
The v3 diagnostic finds matching logical environment values but absent initial proc-environment keys in actual TP workers.
The v4 intervention sets documented SPT_NOENV=1 before child startup.
It changes process-title storage, not the model, gate, precision, or lease predicates.
Both successful jobs retain all strict lease checks.
The [source review](setproctitle_noenv_source_review_v1.md) records the CPU red/green seam and initialization limits.

## Interpretation for discovery

These controls strengthen the same-resource comparison set.
They establish neither a new parallelism axis nor an adaptive distributed runtime.
They do not compare the protected CDB factorial study.
They do not establish natural quality, strong scaling on a broad request population, or a new optimum at every workload.
Any candidate must still compare matched useful work, state, quality, memory, and timing boundaries.

## Reproduction

Use scripts/native_tp2_dp2_cohort_control.py under the scoped GPU guard with a new output name.
The frozen hybrid protocol is artifacts/native_tp2_dp2_cohort_protocol_v1.json.
The new analyzer is scripts/analyze_native_hybrid_cohorts.py.
Its five CPU tests check denominator, timestamp, trace-lane, idle-group, and identity failures.
Run it with a new output directory.
It also reuses the existing native TP/DP raw-record audit.
The successful guard records check worker cleanup and burner restoration.

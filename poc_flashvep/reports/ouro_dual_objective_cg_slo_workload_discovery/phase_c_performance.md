# Frozen workload transfer and serial confirmation

Performance promotion result: NO_GO_INCREMENTAL for the tested O6 formulation.
The completed public accuracy pilot has identical Joint/G-only correctness labels on 114 questions.
See `quality.md` for caps, partial token parity, paired intervals, and the limits of that pilot.
The parent NO_GO_INCREMENTAL result remains unchanged.

## Scope and clocks

The main completed program contains 888 cohorts and 7,200 offered requests across development, audits, and FINAL.
Later supplementary budget diagnostics remain separate and do not increase the main FINAL sample count.
FINAL itself contains 312 cohorts and 3,744 offered requests.
Thirty jobs run up to four independent measurement workers.
Six additional jobs run one measurement worker at a time.
No recorded serving window overlaps another measurement GPU in all 42 serial policy cohorts.
Existing GPU0–3 background remains unchanged. This is not a fully idle eight-GPU host.

The selected objective is O6 with weight 2 and a120ms single-quantum budget.
All matched adaptive controls share fairness, telemetry, cost information, and one-prefill-span packing.
G-only means adaptive G with a fixed configured C. It does not mean G=1.
The primary comparison uses configured C2048 and an additional C1024 control.
The best tested fixed point is staged G4/C512. Tuned Mixed uses C512.

The primary SLO requires TTFT≤5s and every delivered ITL≤200ms, with an actual complete EOS/chat stop.
This is a request-level condition, not a pooled-token P99 condition.
Goodput below is successful complete requests divided by the drained cohort wall time.
It is not a qualified sustainable arrival rate.
Inputs are pretokenized and arrivals occur inside the worker process, without an HTTP server.

## Actual C movement

O6 Joint changes actual C relative to fixed-C2048 on 61/532 fresh decisions, or11.466%.
The serial MIXED fraction is 8/44, or18.182%.
The serial8K fraction is 5/50, or10.000%.
These are actual legal span choices, not nominal configuration changes hidden by clipping.
There are no fallback-selected C changes in these 61 decisions.

The new choices are concentrated in small spans:

| Changed actual C | Decisions |
| --- | ---: |
|128 |2 |
|256 |49 |
|1024 |10 |

Remaining prompt is≤512 tokens in 51/61 changed decisions and 513–2048 in 10/61.
No C change occurs when remaining prompt exceeds2048 tokens.
Nine changes occur with no active decode. Their decode-risk term is zero.
Those cases therefore cannot be evidence that active decode risk caused the C change.
All 61 selected shapes have empirical lookup support.
Every corresponding fixed-C alternative at the same G lacks exact lookup support:61/61.
The selected modeled E exceeds that alternative in 15/61 cases.
The selected modeled R is lower in 52/61 cases, the 52 decisions with active decode.
This audit identifies a cost-support asymmetry, not a measured physical benefit.
The inherited enclosing/inflation rule was frozen before this study. No new penalty forces C movement.
Lookup discontinuities remain a plausible contributor to the new choices.
They do not prove that an unchosen clipped shape is physically slower.

Across all 532 fresh decisions, planned G counts are G1:350, G2:7, G3:22, and G4:153.
These counts describe planned recurrent groups. Prelude still executes G=0.
All observed prompt/decode work remains Full R4.

## Serial lane: measured latency and goodput

Each row uses three independent process starts and 36 offered requests.
Latency entries are means of cohort quantiles, not a pooled quantile over36 requests.
All requests in these serial cohorts reach a registered complete stop. No response is capped.

| Workload | Policy | TTFT P95 ms | ITL P99 ms | Complete joint attainment % | Drained SLO goodput req/s |
| --- | --- | ---: | ---: | ---: | ---: |
| MIXED | O6 Joint |533.25 |62.81 |100.00 |0.7894 |
| MIXED | O6 G-only C2048 |347.74 |58.86 |100.00 |0.7957 |
| MIXED | O6 G-only C1024 |492.21 |53.99 |100.00 |0.7874 |
| MIXED | O6 C-only |846.86 |117.94 |100.00 |0.7733 |
| MIXED | Best tested fixed |525.60 |120.93 |100.00 |0.7863 |
| MIXED | Stock Mixed |416.18 |101.12 |100.00 |0.7864 |
| MIXED | Completion-fair Mixed |718.71 |78.08 |100.00 |0.7114 |
|8K stress | O6 Joint |1664.08 |71.68 |100.00 |0.8941 |
|8K stress | O6 G-only C2048 |1491.84 |71.56 |100.00 |0.8996 |
|8K stress | O6 G-only C1024 |1802.83 |63.59 |100.00 |0.8884 |
|8K stress | O6 C-only |1639.28 |126.60 |94.44 |0.8353 |
|8K stress | Best tested fixed |1090.08 |119.94 |100.00 |0.8992 |
|8K stress | Stock Mixed |813.65 |110.15 |100.00 |0.8994 |
|8K stress | Completion-fair Mixed |1550.62 |91.27 |100.00 |0.8711 |

Joint does not improve the serial TTFT/ITL frontier against the stronger G2048 control.
Its MIXED TTFT P95 increases185.51ms, with paired95% interval[96.36,293.16]ms.
Its8K TTFT P95 increases172.24ms, with interval[17.56,473.89]ms.
Each interval bootstraps three process-start means.
Small process counts and repeated workload families limit population inference.

## Paired goodput comparisons

The ratio is Joint goodput divided by control goodput.
The interval describes the paired goodput difference in requests/s, not the ratio.
Intervals are pointwise exploratory intervals, not multiplicity-adjusted family-wise guarantees.

| Workload | Control | Ratio | Joint minus control req/s | Paired95% interval req/s |
| --- | --- | ---: | ---: | --- |
| MIXED | G2048 |0.9921× |-0.006287 |[-0.007338,-0.004877] |
| MIXED | G1024 |1.0025× |0.001956 |[-0.002945,0.006959] |
| MIXED | Best fixed |1.0039× |0.003032 |[-0.013221,0.012213] |
| MIXED | Stock Mixed |1.0038× |0.002962 |[-0.005530,0.017468] |
| MIXED | Fair Mixed |1.1096× |0.077976 |[0.002372,0.227883] |
|8K stress | G2048 |0.9939× |-0.005479 |[-0.021866,0.005378] |
|8K stress | G1024 |1.0065× |0.005742 |[0.000559,0.015162] |
|8K stress | Best fixed |0.9944× |-0.005061 |[-0.017492,0.002135] |
|8K stress | Stock Mixed |0.9942× |-0.005235 |[-0.031009,0.014763] |
|8K stress | Fair Mixed |1.0265× |0.023048 |[-0.008825,0.077526] |

A point gain over fair Mixed alone is not an incremental Joint result.
The matched G2048 control remains as good or better.
The stock Mixed control is also strong and has lower TTFT in both serial workloads.
Native Mixed has different co-packing capability, so its differences cannot isolate the C dimension.

## Output-work confounding

Joint/G2048 token sequences match on 35/36 MIXED requests and 34/36 stress requests.
Their ordered per-request prompt plans match on 28/36 and 29/36, respectively.
The comparison therefore contains real schedule differences and some different natural outputs.
The observed goodput ratios are not perfectly equal-work causal estimates.

Two of three MIXED cohorts have identical output sequences for every request.
Their postselected equal-output throughput ratio averages0.9919×.
One of three stress cohorts has complete output equality. Its ratio is 1.0070×.
These subsets are selected after execution and cannot establish a population gain.
They do not reveal the registered≥5% incremental advantage.

## Transfer across five workloads

The four-worker characterization preserves both arrival parameters and all five workloads.
Against G2048, Joint point goodput ratios range from 0.6379× to1.0325×.
The low LONG ratio has a wide interval and capped natural responses.
It is not proof of a universal performance loss.
CHAT and BURSTY show no confidence-aware≥5% incremental gain over the stronger G controls.
Apparent gains against fair Mixed disappear as a Joint-specific claim once G-only is considered.

For Joint, actual prompt/output P10/P50/P90 are:

| Workload | Observed requests | Prompt tokens | Output tokens | Caps |
| --- | ---: | --- | --- | ---: |
| CHAT |72 |217/267.5/347.8 |28.2/109/198.7 |0 |
| MIXED, both lanes |108 |216.4/289/1855.5 |32/147.5/235 |0 |
| LONG |72 |1409.9/1812.5/2321.5 |142/181/297.1 |2 |
| BURSTY |72 |218.2/290.5/1926.3 |32.1/150/215.2 |0 |
|8K stress, both lanes |108 |213.4/267/365.2 |26.8/126/220.1 |0 |

The stress mixture contains one long request and 11 short requests per cohort.
Its overall P90 is therefore not an8K context statistic.
The delayed long-request subgroup must be inspected separately in the raw manifest.
These descriptive length summaries reuse source variants across registered conditions.
They are not independent request samples for a confidence interval.

## Capacity decision

Qualified sustainable capacity is NOT_RUN.
Each cell has only 36 offered requests per arm, not the required1,000.
The registered incremental performance gate fails even when real C movement exists.
A larger capacity sweep would not resolve that missing matched benefit.
The study therefore stops before a large arrival-rate matrix or a more complex optimizer.
No finite drained goodput above is relabeled as sustainable serving capacity.

Data: `main_table.csv`, `paired_comparisons.csv`, `final_primary_gate_evidence.csv`, `Joint_C_change_summary.json`, and `whole_cohort_output_parity.csv`.
Postselected diagnostics: `same_output_cohort_summary.csv`.
Plots: FiguresD/E/F/G/H/I in `figures/`.

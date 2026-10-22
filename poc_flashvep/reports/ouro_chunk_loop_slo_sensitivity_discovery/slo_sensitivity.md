# SLO sensitivity and serving-capacity limits

## Contract

The primary reference is warmed GPU4-alone decode at P2033 and resident N4.
The reference is the median of three cohort ITL medians: 27.189323 ms.
It is not a 4K/B32 reference.

| Tier | TBT multiplier | Every-token limit, ms |
| --- | --- | --- |
| Strict | 5 | 135.946614 |
| Moderate | 10 | 271.893227 |
| Relaxed | 25 | 679.733069 |

Absolute sensitivity limits are 75/100/125/150/200/250/500 ms.
Fixed TTFT limits are 2/5/10 s.
The primary grid uses 2/5 s TTFT and all three relative TBT tiers.
The registration predates held-out policy measurements.

The prompt-bin TTFT registration also contains 1.25×/1.5×/2× calibration limits.
`calibrated_TTFT_sensitivity.csv` applies those limits to matching 2K/8K article bins.
Conversation prompts have no matched calibration bin.
The calibration uses historical stress prompts, not the same article contents.
This subset analysis is sensitivity evidence, not a universal service requirement.

## Different definitions can select different policies

| Metric | Denominator or rule |
| --- | --- |
| TTFT attainment | Offered requests with a valid first token before the TTFT deadline |
| Observed TBT attainment | Completed requests whose every observed ITL meets the TBT limit |
| Joint natural-response attainment | TTFT and observed TBT pass, with a complete natural response |
| Population P99 TBT | P99 over observed token intervals. This is not a per-request maximum |
| Declared-drain SLO goodput | Qualifying offered requests divided by the arrival-window duration |
| Completed-in-window SLO goodput | Qualifying completions before arrival-window end divided by that duration |
| Drained throughput | Completed work divided by the actual first-arrival-to-finalization duration |

A length cap does not qualify as complete natural response.
A natural joint pass also requires at least two observed output tokens.
Errors and timeouts remain in the offered denominator.
The TBT-only column does not certify complete-answer quality.
A policy can pass pooled P99 while many requests violate their maximum-ITL limit.
These definitions follow different evaluation goals.
See the direct sources in `literature_slo_contract.md`.

## Actual open-loop experiment

Poisson arrivals use 90-second windows and a declared 120-second drain.
The configured rates are 0.125/0.25/0.5/1.0 requests per second per independent GPU.
There are four process starts per rate and eight policies per start.
The burst sensitivity uses groups of four arrivals, 64-second windows, and three starts per rate.
Its configured average rates are 0.25/0.5 requests per second.

The target mixture contains 50% conversation proxies and 25% each 2K/8K article excerpts.
Finite request counts can produce small deviations from those fractions.
The prompts and arrival seeds are common across policies.
Alpaca provides the conversation proxies. They are not ShareGPT traces.
The frozen held-out pool contains 80 conversation prompts and 29/13 article excerpts at 2K/8K respectively.
Those excerpts share 29 source articles. The held-out pool therefore contains 109 source families.
The input manifest retains source rows and content hashes.

Every run uses actual greedy natural generation with native EOS and observed chat-turn stop.
The maximum output is 512 tokens.
Natural response length and content can differ across policies.
The latency comparison therefore does not isolate equal-output scheduling work.

## Statistical interpretation

`serving_tables.md` contains the measured primary moderate-SLO table and latency distributions.
`slo_attainment_by_rate.csv` contains the full frozen grid.
`serving_paired_differences.csv` contains paired differences in percentage points.

The 95% intervals use 5,000 bootstrap draws with seed 20261023.
The bootstrap resamples engine blocks and source families with crossed weights.
Two excerpt lengths from the same article share a source family.
Repeated requests do not make tokens independent statistical samples.
Each arm/rate contains fewer than 1,000 offered requests.
Request-tail P99 and production capacity remain unqualified.

## Capacity rule

The exploratory screen requires no errors or timeouts.
It also requires queued-backlog slope ≤0.05 requests/s and at most 10% unfinished at arrival-window end.
A positive queue slope above the bound is an overload signal.
An unfinished in-service tail can fail the screen without proving queue divergence.
Both reasons remain separate in the data.

A qualified capacity point also requires the target attainment, adequate requests, and a confidence-aware bracket.
The primary target is 90%. Sensitivity targets are 80% and 95%.
No current rate qualifies production capacity under that rule.
An apparent higher passing point is not a measured sustainable-capacity speedup.

Figure D separates TTFT-only, observed TBT-only, and joint attainment.
Figure F identifies the best observed fixed menu item for each displayed SLO cell.
That post-hoc selection is not an online controller or a validated oracle speedup.
Figure G separates exploratory rate screens from declared-drain goodput.

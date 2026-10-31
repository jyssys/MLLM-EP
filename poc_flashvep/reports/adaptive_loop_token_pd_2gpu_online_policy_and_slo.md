# Two-GPU Loop-versus-Token PD: Online Policy and SLO Contract

Status: measured outcomes are attached below. Unmeasured cells remain explicit.
The implementation starts from source commit `b47d8e01e` and separately archived overlays.
Each run records its actual source archive and worker startup hashes.
The final release will identify the publication commit separately.

## 1. What is online

The coordinator selects a provisional partition for each arriving request.
The decision uses only requests already present in the serving runtime.
It does not use future arrivals, actual future output length, or later candidate latency.
It records the selected cut, legal token prefix, P backlog, Decode count, and Decode age.
The log also records reserved KV credit and outstanding D continuations.
The current rule does not use every logged telemetry field as a decision variable.

The rule is the same for Loop and Token decisions.
Let `a` be the calibration median of host TTFT divided by input-token count.
For a new prompt of `L` tokens, the estimated full service cost is `a × L`.
This estimate includes calibration queueing and host execution costs.
It is not a pure GPU-time predictor or a validated queueing model.

The current Decode age is the maximum time since an active request's last observed token.
The heuristic slack is `TPOT_SLO − maximum Decode age`.
The temporary allowance is half the nonnegative slack.
The allowed D-work fraction is that allowance divided by the estimated full service cost.
The rule caps that fraction at 0.75.
If P has no backlog, the rule selects Full-PD.
If D has no active Decode and P has backlog, the fraction is at least 0.5.

The coordinator selects the smallest legal P-side partition that fits this fraction.
Loop candidates are Ouro `0/1/2/3/4` and Huginn `0/8/16/24/28/32`.
Token candidates use the same normalized fractions before chunk alignment.
Huginn therefore includes the `28/32` fraction on both axes.
Legal token prefixes can collapse after alignment to a 256-token boundary.
That quantization is part of the tested token-split contract.

The 0.75 cap means this online rule does not select D-local Full Prefill.
Cut0 remains an independently tested static endpoint.
This is a conservative online prototype, not an exhaustive dynamic optimizer.
Using a TPOT threshold as a token-age heuristic does not prove request-average TPOT safety.
The experiment must measure actual SLO attainment and expose all fallback decisions.

## 2. Admission recheck and ownership

D reserves the complete request KV budget before P starts.
P stops at its provisional loop or token frontier without exporting a snapshot.
The coordinator then checks the current Decode age and remaining-work estimate again.
If the recheck passes, P exports the valid snapshot.
If the recheck fails, P retains all state and completes full Prefill locally.
Token fallback also moves the request to the suffix Prelude queue.
The source does not discard or recompute the completed prefix.
Loop fallback resumes from the next legal recurrent depth.

No D consumer can activate before every transfer completion is present.
The final commit carries the actual cut and token-prefix boundary.
Acknowledgement protects the source buffer lifetime.
All Decode tokens subsequently execute Ouro R4 or Huginn R32 on D.
An acknowledged migration does not authorize early exit or skipped model work.

## 3. Static tuning

The static tuning data is disjoint from all held-out main prompts.
Each candidate uses four calibration requests per model and workload.
The initial loop grid is `0–4` for Ouro and `0/8/16/24/28/32` for Huginn.
The token grid uses the corresponding normalized fractions.
If the baseline runtime projection requires a reduced grid, both axes use `0/0.5/1` fractions.
That grid decision occurs before held-out candidate results.

The selection first maximizes the request-level joint SLO pass fraction.
It then maximizes finite-cohort goodput.
Mean E2E resolves remaining ties.
This tuning budget does not prove a globally optimal static configuration.
The study must retain every tested calibration point, including rejected choices.

## 4. SLO calibration

The baseline is native-streaming PD_G1 on disjoint complete ShareGPT prompts.
Each model and workload uses eight controlled-generation requests.
The TTFT threshold is 1.5 times the calibration request TTFT p95.
The TPOT threshold is 1.5 times the calibration request-average TPOT p95.
These factors were declared before candidate serving results.

Calibration arrivals are separated by 2 ms.
This is saturated diagnostic calibration, not the Poisson main workload.
W3 calibration uses the normal branch of its calibration manifest.
Held-out W3 then tests the declared long-input burst against that frozen service target.
The calibration file records the actual prompt and output lengths.
Neither the eight-request quantiles nor their derived thresholds establish population tail behavior.

The initial offered rate is 0.8 times baseline cohort completion throughput.
W2 uses factor 1.0. W4 uses factor 1.1.
The W3 long-input segment uses five times its background rate.
All arms share the same model-specific arrival timestamps and prompt order.
Later rate sweeps must retain these SLO thresholds.

## 5. Metric definitions

A request passes the primary SLO only if it completes and satisfies both thresholds.
TTFT starts at its scheduled open-loop arrival, not its later harness admission.
Request-average TPOT is `(last token time − first token time) / (output tokens − 1)`.
Failures and missing requests remain in the offered-request denominator.
Controlled output targets are not Natural-EOS lengths.

Finite-cohort SLO goodput is passing requests divided by arrival-to-final-drain wall time.
Completed throughput and output-token throughput use that same wall-time denominator.
The offered arrival rate is reported separately.
These quantities are not interchangeable during cohort fill and drain.

ITL uses observed consecutive output-delivery timestamps.
Multi-token delivery events are counted separately because they lack individual token timestamps.
Request-average TPOT and token-population ITL quantiles remain separate columns.
Per-request maximum ITL is not the primary TPOT SLO.

One 24–32-request start is exploratory.
Wilson intervals are descriptive request-count intervals, not evidence of independent token trials.
Qualifying paired uncertainty requires independent server starts.
A finite passing rate point is not an unbounded sustainable capacity result.
Queue growth and final drain remain explicit evidence.

## 6. Causal comparison limits

Native PD_G1 streams completed full-depth token chunks.
The Loop and Token methods use snapshot handoff.
Full-cut snapshot controls use the same W1 requests, seeds, and arrivals as native PD_G1.
A transport-only improvement is not a Loop-partitioning improvement.

Each workload/model uses one physical GPU pair for all seven arms.
Pairs cross between workloads and later independent starts.
Clean runs exclude other experiments and approved burners across the shared host GPUs.
Burners remain off during consecutive clean cells to avoid startup interference.
The resource manager restores them after the measured phase.

No main arm uses shared-GEMM Prefill/Decode Mixed co-batching.
These arms therefore do not represent fully optimized production Mixed serving.
Generic logical-layer equivalence remains an unresolved novelty limit from the prior analysis.
This campaign does not implement or evaluate Layer Split.
Measured Loop-versus-Token gains alone cannot prove Loop-exclusive novelty.

The following section summarizes the preserved online choices and measured comparisons.

## Final measured outcomes

| Model | Workload | Arm | Completed N | D continuations | P-full fallbacks | Actual boundary/count |
| --- | --- | --- | --- | --- | --- | --- |
| ouro | W1 | Loop_Dynamic | 32 | 0 | 0 | {"4": 32} |
| ouro | W1 | Token_Dynamic | 32 | 0 | 0 | {"1046": 1, "1079": 1, "1150": 1, "1221": 1, "1303": 1, "1357": 1, "1429": 1, "1446": 1, "1565": 1, "1585": 1, "1615": 1, "1654": 1, "1670": 2, "1672": 1, "1676": 1, "1688": 1, "1728": 1, "1804": 1, "1810": 1, "1812": 1, "3042": 1, "3155": 1, "3164": 1, "3211": 1, "3221": 1, "3257": 1, "3417": 1, "3470": 1, "961": 1, "993": 2} |
| ouro | W2 | Loop_Dynamic | 24 | 5 | 1 | {"2": 5, "4": 19} |
| ouro | W2 | Token_Dynamic | 24 | 5 | 1 | {"2048": 2, "2071": 1, "2083": 1, "2188": 1, "2280": 1, "2304": 3, "3030": 1, "3107": 1, "3158": 1, "3164": 1, "3165": 1, "3171": 1, "3200": 1, "3211": 1, "3231": 1, "3257": 1, "3284": 1, "3305": 1, "3417": 1, "3470": 1, "3472": 1} |
| ouro | W3 | Loop_Dynamic | 32 | 0 | 0 | {"4": 32} |
| ouro | W3 | Token_Dynamic | 32 | 0 | 0 | {"1014": 1, "1046": 1, "1079": 1, "1150": 1, "1221": 1, "1303": 1, "1357": 1, "1429": 1, "1446": 1, "1565": 1, "1615": 1, "1654": 1, "1670": 2, "1672": 1, "1688": 1, "1728": 1, "1804": 1, "1810": 1, "1812": 1, "3042": 1, "3155": 1, "3164": 1, "3211": 1, "3221": 1, "3257": 1, "3417": 1, "3470": 1, "868": 1, "961": 1, "993": 2} |
| ouro | W4 | Loop_Dynamic | 24 | 0 | 0 | {"4": 24} |
| ouro | W4 | Token_Dynamic | 24 | 0 | 0 | {"1014": 1, "1046": 1, "1079": 1, "1118": 1, "1150": 1, "1180": 1, "1186": 1, "1196": 1, "1221": 1, "1303": 1, "1357": 1, "1429": 1, "1446": 1, "815": 1, "868": 1, "900": 1, "931": 1, "934": 1, "952": 1, "961": 1, "970": 1, "975": 1, "993": 2} |
| huginn | W1 | Loop_Dynamic | 32 | 0 | 0 | {"32": 32} |
| huginn | W1 | Token_Dynamic | 32 | 0 | 0 | {"1002": 1, "1050": 1, "1150": 1, "1160": 1, "1168": 1, "1288": 1, "1369": 1, "1410": 1, "1440": 1, "1517": 1, "1554": 1, "1583": 1, "1604": 1, "1609": 1, "1638": 1, "1642": 1, "1646": 1, "1677": 1, "1678": 1, "1812": 1, "1828": 1, "1857": 1, "3005": 1, "3011": 1, "3059": 1, "3110": 1, "3143": 1, "3294": 1, "3295": 1, "3374": 1, "968": 1, "975": 1} |
| huginn | W2 | Loop_Dynamic | 24 | 2 | 0 | {"16": 2, "32": 22} |
| huginn | W2 | Token_Dynamic | 24 | 2 | 0 | {"2014": 1, "2048": 1, "2097": 1, "2108": 1, "2304": 1, "2516": 1, "3005": 1, "3011": 1, "3019": 1, "3032": 1, "3035": 1, "3046": 1, "3083": 1, "3110": 1, "3143": 1, "3148": 1, "3169": 1, "3206": 1, "3294": 1, "3295": 1, "3301": 1, "3360": 1, "3374": 1, "3420": 1} |
| huginn | W3 | Loop_Dynamic | 32 | 0 | 0 | {"32": 32} |
| huginn | W3 | Token_Dynamic | 32 | 0 | 0 | {"1001": 1, "1002": 1, "1050": 1, "1150": 1, "1160": 1, "1168": 1, "1288": 1, "1369": 1, "1410": 1, "1440": 1, "1554": 1, "1583": 1, "1604": 1, "1638": 1, "1642": 1, "1646": 1, "1677": 1, "1678": 1, "1812": 1, "1828": 1, "1857": 1, "3005": 1, "3011": 1, "3059": 1, "3110": 1, "3143": 1, "3294": 1, "3295": 1, "3374": 1, "863": 1, "968": 1, "975": 1} |
| huginn | W4 | Loop_Dynamic | 24 | 0 | 0 | {"32": 24} |
| huginn | W4 | Token_Dynamic | 24 | 0 | 0 | {"1001": 1, "1002": 1, "1031": 1, "1050": 1, "1083": 1, "1122": 1, "1150": 1, "1155": 1, "1160": 1, "1168": 1, "1172": 1, "1188": 1, "1288": 1, "1369": 1, "1410": 1, "1440": 1, "863": 1, "877": 1, "914": 1, "919": 1, "968": 1, "975": 1, "976": 1, "978": 1} |

Token-prefix values can change only because prompt length and legal alignment change.
Boundary variation alone is not evidence of adaptive benefit.
The main comparison CSV separates SLO-goodput ratios from p95 E2E speedups.
The paired-start CSV contains only equal-rate repeated comparisons.
Stable capacity remains NOT_ESTABLISHED for this finite experiment.

## Why the online policy selects Full-PD

| Model | Workload | Arm | Initial decision reason | N |
| --- | --- | --- | --- | --- |
| huginn | W1 | Loop_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 6 |
| huginn | W1 | Loop_Dynamic | NO_P_BACKLOG | 26 |
| huginn | W1 | Token_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 6 |
| huginn | W1 | Token_Dynamic | NO_P_BACKLOG | 26 |
| huginn | W2 | Loop_Dynamic | D_IDLE_HALF_WORK_FLOOR | 2 |
| huginn | W2 | Loop_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 18 |
| huginn | W2 | Loop_Dynamic | NO_P_BACKLOG | 4 |
| huginn | W2 | Token_Dynamic | D_IDLE_HALF_WORK_FLOOR | 2 |
| huginn | W2 | Token_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 15 |
| huginn | W2 | Token_Dynamic | NO_P_BACKLOG | 7 |
| huginn | W3 | Loop_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 22 |
| huginn | W3 | Loop_Dynamic | NO_P_BACKLOG | 10 |
| huginn | W3 | Token_Dynamic | D_AGE_SLACK_EXHAUSTED | 1 |
| huginn | W3 | Token_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 21 |
| huginn | W3 | Token_Dynamic | NO_P_BACKLOG | 10 |
| huginn | W4 | Loop_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 11 |
| huginn | W4 | Loop_Dynamic | NO_P_BACKLOG | 13 |
| huginn | W4 | Token_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 14 |
| huginn | W4 | Token_Dynamic | NO_P_BACKLOG | 10 |
| ouro | W1 | Loop_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 6 |
| ouro | W1 | Loop_Dynamic | NO_P_BACKLOG | 26 |
| ouro | W1 | Token_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 6 |
| ouro | W1 | Token_Dynamic | NO_P_BACKLOG | 26 |
| ouro | W2 | Loop_Dynamic | D_IDLE_HALF_WORK_FLOOR | 6 |
| ouro | W2 | Loop_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 9 |
| ouro | W2 | Loop_Dynamic | NO_P_BACKLOG | 9 |
| ouro | W2 | Token_Dynamic | D_AGE_SLACK_EXHAUSTED | 1 |
| ouro | W2 | Token_Dynamic | D_IDLE_HALF_WORK_FLOOR | 6 |
| ouro | W2 | Token_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 8 |
| ouro | W2 | Token_Dynamic | NO_P_BACKLOG | 9 |
| ouro | W3 | Loop_Dynamic | D_AGE_SLACK_EXHAUSTED | 2 |
| ouro | W3 | Loop_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 20 |
| ouro | W3 | Loop_Dynamic | NO_P_BACKLOG | 10 |
| ouro | W3 | Token_Dynamic | D_AGE_SLACK_EXHAUSTED | 4 |
| ouro | W3 | Token_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 18 |
| ouro | W3 | Token_Dynamic | NO_P_BACKLOG | 10 |
| ouro | W4 | Loop_Dynamic | D_AGE_SLACK_EXHAUSTED | 1 |
| ouro | W4 | Loop_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 6 |
| ouro | W4 | Loop_Dynamic | NO_P_BACKLOG | 17 |
| ouro | W4 | Token_Dynamic | MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE | 6 |
| ouro | W4 | Token_Dynamic | NO_P_BACKLOG | 18 |

Evidence class: deterministic reconstruction from the recorded online state. It is not a counterfactual GPU measurement.
The cost coefficient includes calibration queueing. It is not exclusive Prefill GPU time.
The policy also compares continuation cost with next-token age slack, although the primary SLO uses request-average TPOT.
This conservative restriction can exclude partitions that a different policy could consider.
A Full-PD decision does not prove that every legal dynamic partition lacks headroom.
No threshold or execution rule changed after held-out serving results.

The P-backlog observation includes requests waiting for D reservation.
It is not a direct Prefill-only GPU queue measurement.

## Source and verification

Measured-data release: `d8a0209a0d5ab205308db798b9f04c6b413ad327`.
The primary Loop policy used a D continuation for 5/112 Ouro requests and 2/112 Huginn requests.
All seven cases occurred in W2.
Every repeated W1 Dynamic trial used full Prefill on P.
Its timing differences therefore do not demonstrate adaptive recurrent partitioning.

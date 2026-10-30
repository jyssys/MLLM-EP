# Two-GPU Loop-versus-Token PD: Online Policy and SLO Contract

Status: IN_PROGRESS. Measured serving comparisons are PENDING.
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

Final online decisions, gain tables, rate sweeps, and model verdicts are PENDING.

# Deadline and measurement contract

The primary clock is the in-process monotonic client clock.
TTFT starts at scheduled arrival, not admission.
Each observed output token receives its own timestamp.
Recurrent-loop completion is not output-token completion.

For an unfinished decode request, the next deadline equals its last observed token time plus the registered TBT limit.
For a request without output, the first deadline equals scheduled arrival plus the TTFT limit.
A stopped request has no future-token deadline.
The scheduler cannot inspect future arrivals, output length, output IDs, or future execution durations.

Candidate feasibility includes the proposed prefill quantum and the remaining decode path.
Static R4 can require several more recurrent passes and coda/sampling.
The stock fairness counter resets on a recurrent batch, not an emitted token.
Therefore one decode scheduling turn does not establish deadline safety.

Native `loops_done` is a logical issued frontier.
It can advance before the corresponding GPU event completes.
Consumers remain event-ordered in RLT.
A deadline model must also account for pending submitted GPU work.
It cannot interpret every issued loop as physically complete.
The cost observer records pending submission-event count in newer rows.
The full-token target includes the actual observed queue and completion delay.
An online readiness test must use causal event status, not future measured completion time.

C can change only at a new safe chunk boundary.
An existing staged chunk retains its token span.
G can change only at a legal recurrent frontier.
The actual group is clipped to the remaining depth.
Each request initially retains at most one staged chunk.

CUDA-event time, aligned device completion, host submission time, and client ITL remain separate fields.
Diagnostic CUDA-event runs do not supply headline clean serving latency.
Completion labels can use later timestamps during model training or analysis.
Those future labels must never enter online predictor features.

SLO thresholds remain prospective until reference calibration completes.
The absolute sensitivity grid preserves 75/100/125/150/200/250/500 ms TBT limits.
The TTFT grid is 2/5/10 s.
Relative tiers use 5/10/25 times a recorded decode-only reference.
Final numeric tiers must freeze before held-out adaptive comparisons.

Primary joint attainment requires TTFT compliance, every observed TBT within its limit, and natural completion.
Length caps, errors, and timeouts remain in the offered denominator.
Population p99 TBT is a separate metric.
Arrival-window goodput and drained throughput use different time denominators.

Qualified capacity initially requires at least 1,000 offered requests and three independent starts per arm/rate.
Queue stability and confidence-aware 90% attainment must also hold.
Smaller cells remain exploratory.

# Online request-average TPOT budget

## Metric distinction

The frozen TPOT screen measures each request's average interval after its first output.
It is not a hard deadline for every individual token interval.
The frozen ITL p99 limit remains a separate tail constraint.
Both Token and Loop policies use the same observed-state calculation below.
This policy change precedes every Dynamic GPU experiment in this campaign.

## Prediction from observed history

Let `n` be the number of output tokens already delivered.
Let `f` be the observed first-delivery time.
Let `t` be the current time.
Let `b` be the calibrated conservative token-interval prediction.
Let `m` be the unchanged 90% request-average TPOT margin.

The predicted extra-delay budget is:

```text
mean_budget = m * n - (t - f) - b
```

After one more output, the request has `n` token intervals.
The formula assumes that this next output can terminate the request.
It does not assume a future output length or future EOS position.
When no first output exists, the policy retains the conservative budget `m - b`.

For existing Decode requests, the extra work must also satisfy:

```text
extra_cost <= ITL_limit - maximum_observed_token_age - b
extra_cost <= minimum_active_request_mean_budget
```

The policy does not subtract elapsed time as though that time were GPU progress.
It retains the full conservative next-interval prediction `b`.
The handoff frontier repeats both checks.
KV credit and pending-continuation checks remain unchanged.

## Limits

These are finite calibrated predictions, not hard concurrent execution bounds.
Actual request TPOT and token ITL still determine acceptance after execution.
No evaluation threshold changes.
An empty Decode batch has no existing token interval to protect.
It still requires legal reservations and a frontier recheck.

CPU tests distinguish request-average TPOT from the token-tail limit.
They also check that an older request's mean budget cannot override an unsafe ITL prediction.
This is a common scheduling correction, not Loop-specific novelty.

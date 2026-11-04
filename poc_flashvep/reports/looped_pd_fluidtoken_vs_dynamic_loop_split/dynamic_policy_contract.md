# Dynamic policy contract

Status: implementation prepared on CPU. No online GPU result is reported here.

The implementation uses the label `FluidToken-style reproduction`.
It is not the official FluidPD engine or its complete queue-budget algorithm.

## Observed state

The coordinator chooses a cut when native admission reserves the destination's full KV requirement.
It uses the current queue, request ages, active Decode count, committed KV credits, and recent token intervals.
It does not read future arrivals, actual future output lengths, reference answers, or held-out latency labels.
The destination rechecks safety at the actual recurrent/token frontier.
Rejection resumes the existing P-side continuation through the qualified fallback protocol.

## Shared selector

Token and Loop policies use the same pressure and cost rules.
PPI uses cumulative calibrated P service and remaining frozen TTFT budgets.
DPI distinguishes active Decode pressure from committed KV capacity.
The selector admits at most one pending continuation.
It requires a measured cost table with the same execution signature and hashed calibration evidence.
It rejects missing calibration instead of substituting an assumed service time.
The SLO values must exactly match the frozen contract.

The active selector considers known later requests in queue order.
It assigns safe P-service relief to the current earlier request when those later requests have predicted deadline deficits.
It subtracts additional exposed-transfer cost from the predicted P relief.
It selects the smallest qualified action that covers the deficit, or the largest safe relief when the budget is insufficient.
No later deadline deficit means Full-PD. A pending continuation also forces Full-PD.

The earlier request-local completion selector remains a CPU control, not the active runtime policy.
That selector can miss queue benefits when the current request becomes slightly slower but later requests finish earlier.
A synthetic regression test covers this distinction. Its numbers are not experiment results.

Recent request token intervals provide a conservative latency signal, not a GPU-kernel duration.
The calibration must report stream-event enclosures and queue-free host service separately.
Neither value is an exclusive E2E component.
GPU time, transfer time, and queue time must not be added as independent terms when they overlap.

## Differences from FluidToken

The decision occurs at request admission and at the transfer frontier.
It does not reproduce FluidPD's complete per-P-scheduling-iteration redistribution of a shared suffix budget.
The candidate set contains only exact cuts that passed this runtime's qualification.
The same finite candidate budget applies to Token and Loop policies.
An additional token-ITL gate is stricter than a request-average-only slack calculation.
FluidRole is not implemented.

These differences must remain explicit in the final comparison.
The implementation must not be reported as the original FluidPD system.
CPU tests alone do not qualify output correctness, fallback ownership, or serving performance.

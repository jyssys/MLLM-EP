# Compute budget and completed-token stalls

The TRAIN intervention completed all eight budgets: 10/20/40/60/80/120/160/240 ms.
It covers P2033/N4 and P8177/N8 with paired policies on each GPU/process block.
The queue contains 32 jobs and 448 bounded natural observer cohorts.
These are exploratory shared-host results, not quiet serving headlines.

The budget constrains a predicted individual prefill service unit.
It does not constrain the sum of all work before an output token.
Decode still needs its remaining R4 passes, coda, sampling, and host delivery.
Fairness can prevent repeated prefills without making each complete-token gap equal to B.

The observed response is not monotone.
A smaller budget can reduce individual prefill exposure while increasing scheduling events and time to complete a prompt.
At very small budgets, no action in the registered menu may fit the estimate.
The bounded progress fallback can then exceed B and use a small inefficient chunk.
That behavior can worsen TTFT and total output rate.
It must not be presented as successful budget adherence.

At larger budgets, the selector can use larger chunks or longer groups.
This can improve prompt completion while increasing individual stalls.
Native Mixed provides a separate strong alternative through prompt/decode co-batching.
Joint's larger menu is not sufficient evidence that it beats Mixed.

Actual budget overruns are in `budget_overruns.csv`.
The rows separate Atomic, staged prelude, and staged recurrent paths.
The action trace also retains unknown shapes and progress escapes.
The model's held-out upper-estimate exceedance is a different statistic from live configured-budget overrun.

`output_prefill_exposures.csv.gz` separates submitted prefills from GPU service intervals overlapping each output gap.
It preserves the union of aligned intervals rather than adding overlapping intervals twice.
An overlapping mixed interval is not attributed entirely to prefill compute.
These associations do not alone establish a removable critical-path fraction.

The quiet B40/B120 confirmation uses new source families and the corrected implementation.
Its results remain separate from the TRAIN budget-dose curves.
All 12 quiet confirmation starts completed.
At P8177/N8, raising J1's B40 to B120 changed pooled ITL P99 from 96.31 to 172.19 ms.
TTFT P95 changed from 1,193.18 to 1,062.16 ms.
This is a measured TTFT/tail trade-off, not a monotone SLO improvement.
Tuned G-only selected the same 144 per-request plans as Joint over the four confirmation cells.
The measured incremental Joint benefit is therefore absent in those cells.

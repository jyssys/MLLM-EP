# Timing and statistics contract

## Separate clocks

Clean mechanism timing uses `perf_counter` at actual host output delivery.
Each output update must contain exactly one new token.
TTFT and full-answer E2E require separate natural-serving measurement.
They are not inferred from recurrent submission duration.

The primary decode span starts at the latest first output across cohort members.
It ends at the last delivered output.
The numerator counts only actual output tokens delivered strictly inside that span.
The rate equals that numerator divided by the span.
Per-request adjacent output gaps also supply ITL mean/P50/P95/P99.
The ordinary screen can retain prefill-induced recurrent-phase fragmentation after its first-output boundary.
The matched-frontier lane removes that initial-state difference explicitly.

Warm-up and model construction precede clean measurement.
Any new graph preparation after measurement starts remains inside latency.
The hardware screen ignores EOS and caps generation at64 tokens.
Quiet confirmation caps it at128 tokens.
These are equal-length free-running hardware probes, not forced output IDs or complete-answer quality tests.

## Event observer

The observer records CUDA events after native request waits.
The end event follows actual model submission on the selected stream.
It records event/host alignment uncertainty and producer readiness.
Those intervals include service gaps and do not measure pure kernel busy time.
CPU selection, host submission, and CUDA service can overlap.
Their sums must not form an additive E2E breakdown.

Heavy observer results remain separate from clean timing.
The GPU timeline uses measured event intervals and actual host submission timestamps.
It is not a fabricated kernel timeline or a claim of zero rank waiting.
There are no TP ranks or collectives in this single-GPU worker experiment.

## Pairing and uncertainty

Every policy crosses the same physical GPU within each process block.
Policy order rotates across devices and quiet process starts.
Repeated cohorts first reduce to a process-block mean.
The speedup is grouped completed-token rate divided by matched D1 rate.
Percentile intervals use10,000 process-block bootstrap draws.
Quiet confirmation uses three independent starts on each of two GPUs.
The repeated source family limits population generalization.
Intervals are pointwise exploratory intervals without multiplicity correction.
Token and recurrent-kernel counts do not define independent statistical samples.

The eight matched-state interventions are diverse component diagnostics.
They do not establish an arrival-distribution confidence interval.
No drained cohort rate becomes sustainable request capacity.
No complete-answer SLO claim follows from EOS-ignored hardware probes.

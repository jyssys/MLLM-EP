# Huginn Loop28 Decode tail diagnosis

## Scope

Reference commit: `21a067b8bf2242bb8a16fc8c59ed59193e481f54`.
Preserve its original results, SLOs, source files, staged index, and external Runtime.
Do not implement a scheduler fix or publish a new speedup claim.
Keep approved Burners active outside isolated GPU diagnostics.

## Reproduction and hypotheses

The read-only raw-output replay reproduces TTFT/ITL/TPOT ratios of 0.48852794, 1.70878714, and 1.20959613.
These ratios use the three original paired starts.
The protected-tail assertion fails deterministically in less than one second.

The primary four arms are Native-path Zero, full-prompt Loop28, Loop28 with 256-token tiles, and Loop31 with 512-token tiles.
The earlier full-prompt Loop31 event diagnostic supplies secondary evidence.

Ranked hypotheses:

1. Earlier Decode admission changes the Decode batch distribution and part of its cost.
2. Independent prompt work delays same-size Decode work on shared device resources.
3. Host scheduling, cohort retirement, and stream ordering add waiting outside model execution.
4. A cold graph or a missing cache entry accounts for the regression.

The original static cuts do not execute online safe-budget admission.
Copied prediction fields in a configuration are not evidence that the Runtime consumed them.
Distinguish absent admission from an incorrect runtime prediction.

## Existing-evidence analysis

Compare all three held-out starts and the matched selection trace.
Retain the original isolation exclusion and replacement rule.
Count actual D prompt spans and actual Decode batch histograms.
Condition output delivery intervals on observed workload composition.
Do not describe Host live requests as physical GPU batch size.
Use actual CUDA Event metadata for device batch size where available.
Do not assign additive causal percentages to p99 differences.

## Short GPU diagnostics

Use GPUs 4/5 under the existing exclusive quiet-lane guard.
Use the original BF16 model, R32, 2048/16, P/D slots 2/8, KV pools, graph budget, and first 64 original prompts.
Use the same prefix of Arrival Seed 2026110940 for all new N64 cells.
The new short cohort does not replace the frozen N128 measurements.

Run a serialized N8 state check before the diagnostic controls.
Run clean and event-observed N64 controls for each primary arm.
Repeat Zero without events to measure A/A variation.
Add a Loop28 counterfactual that orders device operations through explicit event dependencies.
That counterfactual removes permission for model-operation overlap.
It is not a proposed performance optimization.

Record each D operation's actual stream, CUDA Event endpoints, host submission, and Native ready-test transitions.
Record queued phases and pending CODA outputs without additional readiness queries.
Label operation enclosures separately from individual kernel execution.
Keep physical kernel union, SM utilization, HBM traffic, and cache traffic `NOT_MEASURED` unless measured independently.

Check every generated token against the original Native output prefix cohort.
Check full recurrent work and actual child exit status.
Compare event-observed and clean TTFT, TPOT, ITL p99, and maximum ITL.
Use a five-percent diagnostic sensitivity screen.
A failed screen makes the trace diagnostic-only.
One paired control does not identify the profiler's causal effect beyond A/A variation.

## Interpretation

Report same-batch device costs and observed overlap separately from Host waiting.
Use common-support workload reweighting as a descriptive calculation only.
Label unsupported batch strata and unresolved causal contributions `NOT_IDENTIFIED`.
Do not infer SM, HBM, cache, or launch contention from an enclosure alone.
Do not infer physical D capacity from NVML or Host queue observations.
Report the dominant cause only if evidence separates it from the other causes.

## Time and closure

The diagnostic task starts at 2026-11-11 05:05:19 UTC.
Its operational safety bound is 150 minutes, including restoration and reporting.
The planned GPU work is approximately 20–30 minutes, not a new long serving campaign.
Each GPU cell has its own timeout and Native resource lifetime checks.
Stop a failed owned cell without changing the SLO or the original result.
Restore approved engines on all eight GPUs after diagnostics.
This request does not authorize a new remote publication.

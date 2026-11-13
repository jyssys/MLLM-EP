# Dense AR Dynamic Layer-wise PD: campaign specification

## Objective

First measure the Native PD and Strong Colocated DP2 trade-off.
Then test whether state-aware layer continuation improves the SLO-qualified frontier.
Do not assume that Colocated harms Decode or that Dynamic beats a tuned static fraction.
Keep workload discovery gains separate from policy gains.

Use a new source and evidence namespace.
Preserve the frozen Dense AR implementation and its raw results.
The publication parent is `367d50336791b4c540404053eadf924544a2ca0f`.
The branch is `flashvep/dense-ar-dynamic-layer-pd-crossover-20261113`.

## Exact execution contract

Use Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.
Use BF16, all 36 layers, exact Attention, and greedy sampling.
Do not change model weights, precision, or mathematical dependencies.
P and D retain the full model.
P computes a true prefix and D consumes the transferred Hidden and Residual.
D completes the suffix and then executes Full36 Decode.
The candidate set is Last0/2/4/8/12.
Last12 requires new GPU state and actual-layer qualification.
Original Native keeps the P first token and reads only valid prompt KV.
Source state remains live until its valid ACK.
Destination buffers remain live until their consuming CUDA work completes.

Reuse the exact no-compile BF16 family and Full Decode Graph policy.
Do not reuse invalid Optional-input or compiled boundary results.
Check output tokens, valid KV, final Hidden, actual layer visits, and normal buffer retirement separately.
Use existing tolerances `atol=0.02`, `rtol=0.01` for independent numerical state comparisons.
Never relax them after a failure.

## Workloads and data separation

Start with Input8000 and Output16/32/64/128/200.
Use real ShareGPT/LongBench text with preserved source IDs and hashes.
Use a complete disabled-thinking generation suffix within the 8000-token limit.
Verify the actual tokenizer suffix, not only a configuration label.
Freeze separate qualification, calibration, discovery, and held-out prompt sets.
Use 64 discovery prompts and 128 different held-out prompts.
Do not use repeated prompt IDs as independent data.
Use Ignore EOS for controlled fixed-length evaluation.
Natural EOS is a separate generalization test if the primary gate passes.

Apply identical open-loop Poisson arrival offsets to compared arms.
Record offered, realized, submitted, and completed QPS separately.
Finite-cohort drain rates do not establish sustainable capacity.
Bursty confirmation must use a separate frozen arrival trace.

## Phase A: strong baselines and crossover

Recheck local P capacity and prepared D capacity for batch sizes supported by the KV pool.
Reproduce Native output before migration performance tests.
For Colocated DP2, compare the existing cohort control with actual native chunked-prefill scheduling.
Tune token budgets, per-request thresholds, active sequence limits, and routing on discovery data only.
Keep a configuration only if its own correctness contract passes.
Record failed chunk configurations instead of silently weakening that contract.
If useful, compare round-robin and load-aware routing without future information.

Start QPS near the measured capacity, including underload, transition, and overload.
Adjust the next QPS from the measured bottleneck rather than running a blind full grid.
Record queue ownership, KV refusals, active Decode batch, actual model inputs, and client token timestamps.
Distinguish Host model enclosures, CUDA event enclosures, and physical kernel unions.
Never substitute NVML utilization for physical idle or SM headroom.

A confirmed crossover requires the Native P compute bottleneck and the Colocated Decode trade-off at matched workload and load.
Require independent starts before confirmation.
If no such window appears, report `NO_CONFIRMED_PD_COLOCATED_CROSSOVER_ON_TESTED_GRID`.
Do not weaken Colocated to obtain that window.
Dynamic diagnostics can continue if separate P pressure and extra D-work capacity are verified.
They do not then validate crossover motivation.

## Phase B: pre-registered SLO calibration

Do not evaluate a Dynamic candidate before freezing SLOs.
Use independent low-load Native calibration at each tested output cap.
Target three starts and at least 16 distinct requests per output cap.
Set base TTFT to the largest start-level TTFT p95.
Set base ITL to the largest start-level ITL p99.
Set base TPOT to the largest start-level request TPOT p95.
Round resulting thresholds upward to whole milliseconds.
Use serial low-load calibration so a long output does not create a hidden Decode queue.

| Regime | TTFT factor | ITL factor | TPOT factor |
|---|---:|---:|---:|
| TTFT-priority | 1.25 | 2.00 | 2.00 |
| Balanced | 2.00 | 1.25 | 1.25 |
| Decode-priority | 4.00 | 1.10 | 1.10 |

These factors are research design choices, not claimed production-service standards.
For Input8000, use pre-registered TTFT floors of2000/5000/10000ms for the three regimes, respectively.
Set each TTFT limit to the larger of its calibration product and its regime floor.
These floors express distinct queue-tolerance targets before Dynamic measurements.
They avoid treating every controlled-overload request as automatically useful or automatically failed.
Keep the old SLO as an additional frozen anchor.
Publish calibration inputs and the immutable threshold hash before Dynamic evaluation.
Do not choose a regime from candidate results.
Report all three regimes.

TTFT is arrival-to-first delivered token.
Request TPOT is first-to-last delivery divided by the number of later tokens.
ITL includes the first interval.
Report first ITL and steady ITL separately.
For request Decode pass, require request TPOT and request maximum ITL within their limits.
Joint pass also requires TTFT within its limit.
Request goodput counts joint-passing requests per observation second.
Output goodput counts their generated tokens per observation second.
Report any coalesced delivery intervals explicitly.
Service TTFT remains `NOT_MEASURED` unless the required timing boundary exists.

Absolute pass is separate from Native-relative non-inferiority.
Require ITL p99 and TPOT p95 ratios no greater than1.05.
Use independent starts as bootstrap units.
A point estimate below1.05 does not pass when its confidence bound exceeds1.05.

## Phase C: request-specific dynamic continuation

Choose the cut before submitting the corresponding P prefix.
Do not stop a previously submitted full forward at a past boundary.
A reservation has a unique request generation, cut, expiry, and KV/compute credit.
Before commit, rejection uses the Native fast path.
After commit, retain exact ownership until the suffix and Decode complete.
Never discard a prefix and silently recompute it.

Separate admission feasibility from immediate execution feasibility.
Estimate P cut readiness, all pending D Decode work, P completions, handoff, D suffix cost, and completion waiting.
Use observed history and current state only.
Unknown arrivals require an uncertainty margin.
Use calibrated tail costs, not only mean layer cost.
Do not assume kernel preemption.
Bound continuation waiting and record prediction overruns.
Treat `t2-t1` as an initial estimator, not a measured idle guarantee.

Start with lookup tables and conservative bounds.
Use roofline FLOPs and traffic as structural diagnostics.
Add Attention, KV, launches, graph fallback, transfer, and scheduling costs from measurements.
Do not label theoretical traffic as measured HBM traffic.

## Phase D/E: policy comparison and confirmation

Compare Native, best static cut/fraction, Idle-only, Deadline-aware, and full Cost-aware policies.
Tune static Last2/4/8/12 and fractions25/50/100% on validation data.
Compare state-aware selection with a matched static migration fraction.
Keep the same numerical family and generic runtime optimizations in all arms.
Qualify N8, then actual mixed-cut N32, before N128 evaluation.
Record the actually completed migrated layers, not only accepted offers.

Primary comparison includes Native, Strong Colocated DP2, Best Static, and Dynamic.
Use at least three balanced paired independent starts.
Use held-out prompt IDs and new arrival seeds for final confirmation.
Do not combine selection winners into the confirmation sample.
Extend uncertain non-inferiority with more starts when useful.
Report TTFT, E2E, ITL, TPOT, maximum ITL, exceedances, token/s, joint attainment, and both goodputs.
Show the TTFT–Decode frontier instead of claiming universal dominance.

## Safety, reporting, and publication

Each GPU job has a startup timeout, serving timeout, process-group watchdog, and ownership check.
Use a shared quiet-lane lease for clean timing.
Stop only approved burners under that lease.
Restore burners when the measurement batch finishes or fails.
Keep all eight burners active during CPU-only work.
Verify fresh generation on every GPU at final closeout.

Keep immutable commands, source archives, GPU UUIDs, model revision, seeds, dataset hashes, configurations, and failures.
Use resumable manifests.
Keep profiler controls separate from marker-free primary timing.
An observer that changes required metrics by more than5% is diagnostic only.
Use `MEASURED`, `TRACE_ANALYSIS`, `STRUCTURAL_ACCOUNTING`, `OFFLINE_ORACLE`, `CORRECTNESS_ONLY`, `NOT_MEASURED`, `NOT_IDENTIFIED`, `NOT_RUN`, and `FAILED` explicitly.
Preserve the real checkout, staged index, external runtime, and frozen predecessor files.
Publish related files through a private Git index.
Push the new branch and verify its Remote SHA.

# Known negative space

This document is the exclusion boundary for the night discovery sprint.  A
closed direction may only be revisited when a new causal variable and a fresh
live measurement are both present.

| Direction | Existing evidence | Disposition |
|---|---|---|
| Per-token EP fanout as an online control | 98,880 online MoE rows; Model2→Model3 held-out RMSE did not improve and natural matched-pair sign was unstable | CLOSED |
| Histogram-preserving fanout replay as a method | Operator replay changed latency non-monotonically; online transfer was null | DIAGNOSTIC ONLY |
| Plain expert/rank token imbalance | Load statistics explain ordinary shape but not an orthogonal online signal; TEMPO/DA-MoE/RaMP-adjacent | CLOSED AS NOVELTY |
| Modality-aware TP↔EP switching | Real DeepEP comparison was explained primarily by workload volume; no stable modality crossover | CLOSED |
| Generic dynamic TP↔EP switching | Prior-art crowded (Moebius/HAP) and previous load crossover was not modality-causal | CLOSED |
| Modality-specific MoE granularity | Matched routing replay did not establish a robust modality-specific optimum | CLOSED |
| Simple fragmentation law | Sign changed with M/runtime order; no monotone reusable law | CLOSED |
| Route-complementary request rebatching | Controlled experiments did not deliver robust E2E benefit; ELDR/Semantic Parallelism adjacent | CLOSED |
| Spatial/route-aware chunking | Oracle or live handoff did not survive correctness/runtime overhead | CLOSED |
| Critical-rank assignment coalescing | Trace-driven redundancy did not establish quality-preserving critical-path benefit | CLOSED |
| Naive vision-encoder × DeepEP overlap | Measured negative pairs: dispatch -12.4%, combine -5.0%, expert -8.9% | CLOSED |
| Visual streaming prefill | Image equivalence passed and oracle was 11.4%, but real boundary prototype did not recover a stable qualifying gain | CLOSED |
| Partial/speculative MoE completion | Fresh top-k contribution capture passed only near exact top-8; quality-gated overlap oracle was 0% | CLOSED |
| SpecMoE-like affinity substitution | Output-space surrogates did not make top-4/top-7 safe; substitution itself is prior art | CLOSED |
| Adaptive DeepEP communication SMs | SMS 8/12/20 exact-M lower-envelope median headroom 0.11%, max 2.10% | CLOSED |
| Fixed-shape/history-dependent DeepEP tail method | Root cause was dispatch/event spillover, but direct request-level capped removable share was 1.09%; selective wait failed | CLOSED |
| Always/selective synchronization | Mostly moved waiting, was unstable, or cost throughput; no direct E2E headroom | CLOSED |
| DBO causal wavefront prototype | DBO itself inflated layer work/calls about 5.5× in the tested path; custom wavefront added no benefit | CLOSED |
| Simple DP→EP heterogeneity barrier | One large result reversed across repetitions; source/runtime state confounded and ASAP-adjacent | CLOSED UNLESS NEW DIRECT CONTROL |
| Static max-batched-token tuning | Large same-M difference exists, but one obvious static knob recovers it | TRIVIAL ENGINEERING |
| Simple concurrency reduction | Improves latency by trading away throughput | INVALID COUNTERFACTUAL |
| Warmup/vision-text transition state | One 2× MoE effect disappeared under telemetry-tagged replication and shape-matched warmup | STATE-CONFOUNDED / TRIVIAL |
| Layer heterogeneity alone | Early/mid/late p50 ratios were only about 1.18–1.25× and lacked a stable intervention | LOW HEADROOM |

## Still-open measurement gaps

1. Prior online stage traces used a hook that synchronizes every MoE layer and
   copies routing IDs to CPU; stock DeepEP HT is asynchronous.  The observer
   may have changed the phenomenon.
2. Request-wave labels exist, but exact per-step critical-path mass outside
   MoE (attention, host/model-launch gap, DP coordination) was not captured by
   a low-perturbation observer.
3. In eager DP execution the CPU DP descriptor all-reduce remains, while graph
   mode additionally pads DP ranks to the maximum token count.  The causal
   cost of global request partitioning has not been measured with a trusted
   low-perturbation observer.
4. The installed HT path constructs expert-token metadata from a host list on
   every layer and uses fixed async/layout choices.  Its high-frequency host
   and device critical-path mass has not been isolated.


# Profiling scope and Decode tail gates

## First asynchronous-evidence control

The geometry-4 control completed 512 clean requests with exact background and probe outputs.
Its TPOT p95 was 137.084466287 ms. ITL p99 was 141.779683232 ms.
Maximum ITL remained 1118.597380817 ms. The fix therefore does not remove every large outlier.
All recorded asynchronous evidence was flushed. No writer error was reported.

The matched dual-device capture also completed 512 requests.
Both activity traces passed their matched-wall perturbation checks.
Whole-run TPOT p95 was 139.919848703 ms. Whole-run ITL p99 was 148.305900935 ms.

However, the predeclared 32-second capture interval failed the joint gate:

| Same-interval metric | Measured value | Gate |
|---|---:|---|
| P kernel union | 95.843002764% | PASS, at least 80% |
| P finite service proximity | 107.179492930% | PASS, at least 85% |
| Device-qualified queue half-growth | -0.109375 requests | FAIL, positive growth required |
| D TPOT p95 | 153.750656359 ms | PASS, at most 167.914355323 ms |
| D ITL p99 | 215.016544685 ms | FAIL, at most 192.821253818 ms |
| Minimum D free blocks | 3520 | Retained for the credit check |
| D credit-refusal sample fraction | 49.21875% | PASS under the frozen 50% check |
| Completed exact local probes | 6 | Present in this interval |

The host-ownership queue estimate increased by 2.359375 requests.
The stricter device-qualified point-sampling estimate did not increase.
Both estimates remain in the evidence. The negative result is not relabeled as a pass.

## Interpretation

An acceptable whole-run profiling overhead does not guarantee unchanged latency tails inside the capture interval.
The measured whole-run and same-window SLO results are different contracts.
Both must pass before this campaign declares recovery readiness.
This run is a retained failure, not a source of accepted D-headroom evidence.

## Predeclared fallback

If the dual-device controls fail, repeat the original P-only activity protocol with the corrected evidence writer.
Keep every numerical threshold and all whole-run checks unchanged.
Use exact D-local work, KV credit, and Decode latency from that same P-profiled execution.
Require three independent starts. Do not select a favorable capture window after the run.

A P-only pass cannot provide a measured D kernel timeline.
That panel must say `NOT_MEASURED_IN_ACCEPTED_RUNS`.
Failed dual-device traces cannot fill the missing panel or establish same-run success.

Kernel union is not SM occupancy, achieved FLOPs, or memory bandwidth.
Finite service proximity above 100% shows that the reference is not a universal capacity bound.
These are operational gates, not proof of a physical peak-FLOP limit.

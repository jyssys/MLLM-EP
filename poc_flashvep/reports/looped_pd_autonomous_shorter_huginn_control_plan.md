# Shorter Huginn KV control

## Reason for the new condition

The 2560/16 eight-second profile failed the joint queue and Decode-tail gates.
The 30-second capture stopped response progress after 132 completed requests.
The watchdog terminated that run. Its partial trace is diagnostic evidence only.
Neither result proves a repeatable joint target.

The next condition uses 2048 input tokens and 16 output tokens.
This condition remains inside the user-approved search range.
The shorter prompt reduces per-request KV demand without changing precision or recurrent depth.
It can also reduce Prefill work. Therefore, a higher Prefill-heavy likelihood is a hypothesis, not an assumption.

## Predeclared protocol

1. Compare the original eager path with native Prefill graph reuse at eight requests.
   Check transferred state and full recurrent work.
2. Compare outputs at 32 requests with an independent arrival seed.
3. Calibrate the new input length on disjoint serial requests before held-out serving.
   Use C1024 and full C2048. Keep the tighter per-metric threshold.
   Existing thresholds for all other lengths remain unchanged.
4. Measure P-only finite service with batches one and two, each with three repetitions.
5. Test 24 requests at 1.1 and 1.3 times that measured reference.
   Only a candidate that preserves the frozen Decode limits can enter confirmation.
6. Confirm the selected candidate with 384 offered requests and an eight-second capture.
   Cycle the existing 128 public prompts explicitly. Do not claim 384 independent text samples.
7. Use at most 12 independent 16-token D-local probes, spaced four seconds apart.
   This lower injection frequency follows the preceding tail-latency failure.
   Do not compare this setting with the earlier two-second setting as a matched speedup.
8. Repeat with two additional independent seeds only if the first joint screen passes.

The 80% kernel-union screen, 85% finite-service screen, 5% profiling gate, and positive P-ready growth requirement remain unchanged.
This test does not implement recurrent continuation migration or Dynamic A.

# Failure notes

This file records completed failure investigations. The final CSV tables retain all attempted runs.

## Natural-output gates

Ouro B and PD_G1 match185/200 W1 generated sequences at nominal1request/s.
Huginn B and PD_G1 match193/200 at nominal0.2requests/s.
The controller closes both B performance gates.
Initial40/40 sequential qualification and two-request KV checks do not override these later failures.
The mechanism remains unresolved until separate diagnostics finish.

## Huginn overload at nominal0.8requests/s

Run: `huginn_W1_PD_G1_N2_rate0.8_seed2026103001`.
The engine completes173 of200 planned requests before it raises `PD request/transfer deadline exceeded`.
The remaining27 planned requests count as failures in the SLO denominator.
Only2 of200 planned requests satisfy the frozen joint SLO. The descriptive point attainment is1.0%.

The exception originates in `vllm_rlt/pd/engine.py`, lines433–438.
It checks elapsed request/transfer lifetime against `request_timeout`.
It does not isolate network DMA duration.
The last recorded completion has an E2E latency of293014.75ms.
This is evidence of overload and the existing lifetime deadline, not proof of a communication fault.

The failed run has no completed-drain `window.json`.
The analysis recovers only the planned arrival-window duration from the frozen `arrival_trace.json`.
It does not invent a drain duration, completed-cohort endpoint, or achieved drain-inclusive throughput.
Latency percentiles use the173 completed requests only. They carry a failed-run survivor label.
This run cannot certify sustainable capacity.

The traceback and partial outputs remain unchanged in the run log and raw directory.
The controller does not retry the failed configuration recursively or increase its deadline to improve the result.

## Cleanup warnings

Some completed runs emit UCX CUDA-context warnings after the final request completion.
The final warning inventory preserves the text and its position relative to the last `DONE` line.
These warnings do not automatically become successful request failures or harmless-runtime claims.
Saved terminal status and measured request completion remain separate evidence.

## Order balancing after the B gate closes

The initial three-arm rule rotates order by seed modulo the number of available arms.
After B stops, the remaining two-arm main runs retain the same order because confirmation seeds add1000.
Hardware pairs still change in main confirmation. This does not remove the order limitation.
The later extension alternates order explicitly with `floor(seed/1000) modulo arm_count`.
That change precedes extension execution. It does not alter model computation or the SLO.

## Byte-check scope

The KV fingerprints check transferred valid prompt KV and final hidden state at import.
They do not check every subsequent decode KV write through output token128.
A passing diagnostic cannot exclude every later state error or reopen the failed natural-output gate.

## Primary-population amendment before extension

The first extension draft reduced very-low-rate samples to60–200 arrivals to avoid a long benchmark window.
That draft would change the W1 prompt distribution between arrival rates.
No extension job executed with that draft.
The revised plan preserves all200 source prompts and rounds larger counts to complete200-prompt cycles.
It tests one lower half-rate when PD_G1 has no passing point.
It omits the automatic quarter-rate point instead of substituting a smaller, potentially easier prompt prefix.
The SLO, model, generation settings, and actual initial results remain unchanged.

## Strong colocated control

At the matched Huginn W1 rate0.2 point, plain DP attains96.5% and DP_Mixed attains93.5%.
This is a point observation, not proof of a larger plain-DP capacity.
It also means DP_Mixed cannot be assumed to be the strongest colocated control.
After the primary extension, the plan probes plain DP at0.4 with200 complete-source prompts and independent starts.
It can test an upper point or confirm the lower point if time remains.
The original SLO stays unchanged. Optional4/8GPU extra controls remain lower priority.

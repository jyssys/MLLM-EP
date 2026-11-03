# Longer Huginn capture control

## Evidence before this control

The `window_huginn_s1_nsys384` run completed 384 requests and 24 independent D-local probes.
Its eight-second capture recorded 91.1069% P kernel union.
P service reached 99.5507% of the finite independent reference.
The audited P queue decreased by 2.99265 requests between the observation halves.
Central D ITL p99 was 200.5728 ms. The frozen limit remains 198.3781 ms.
The joint screen therefore failed.
Whole-run profiling overhead was 4.8396%. This does not exclude local profiler disturbance.
All probe outputs and full-depth work accounting passed.

## Ranked hypotheses

1. A short capture samples a transient queue drain after profiler initialization.
   A longer capture should reveal whether backlog growth persists across more service cycles.
2. Extra D-local work creates a narrow tail-latency failure.
   A longer capture can change the estimated percentile but must not change its frozen limit.
3. D admission or native transfer limits the joint interval.
   The longer capture must retain all refusal observations and exclude D-blocked requests from P-ready pressure.

## Controlled test

Keep the first 384-request baseline and probe configuration unchanged.
Change only the P capture duration from eight to 30 seconds.
Keep the same prompt IDs, arrival seed, QPS, probe count, probe interval, and capture-start fraction.
The longer capture is a new immutable run. The failed eight-second result remains part of the final evidence.
Require the original 5% profiling gate, 80% kernel-union gate, 85% service gate, and frozen D SLOs.
Do not select a favorable subsection after measurement.
If this control passes, repeat the 30-second protocol with the two remaining independent seeds.
If it fails, report the failure rather than repeating seeds until one passes.

The probes execute independent full-depth D-local prompts. They are not migrated recurrent continuations.
Kernel union does not measure physical SM saturation or memory bandwidth.

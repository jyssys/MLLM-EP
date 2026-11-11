# Measurement interpretation audit

## Profiler stall watchdog

The first `manual_profile_watchdog.json` uses the phrase "no active kernels".
That phrase is too strong.
The underlying observation is a 0% NVML utilization sample on GPUs 4 and 5.
NVML utilization does not establish an exact kernel-free interval.
The source Python stack remains inside `nixl_cu12._api.transfer`.
The destination Python stack remains inside its scheduler loop.
These observations establish a response-progress stall associated with profiling.
They do not establish the C++ root cause or clean physical idle capacity.
The original watchdog record remains unchanged.

## Graph trace device mapping

The CUDA activity tables use visible device IDs 0 and 1 in these workers.
The worker map resolves them to physical GPUs 4 and 5.
Do not join activity device IDs directly to physical `TARGET_INFO_GPU.id` values.
The published analysis uses the recorded worker UUID and GPU mapping.

## Per-request Event accounting

Every request in a shared batch experiences the full batch enclosure.
The sum across requests can therefore exceed the sum across submissions.
Neither sum is exclusive GPU kernel busy time.
Inter-enclosure gaps can include other requests and transport work.
They are not physical idle windows.

## Async configuration

The initial Async-request candidates resolved to synchronous execution on both Huginn workers.
Their exactness checks do not qualify global Async scheduling.
The separate `fixed_depth_submit` extension does not enable global Async.
It preserves fixed-depth model execution and removes only an unused host Gate readback.

## Native wire-credit and draining-policy confound

The strongest Zero explicitly uses an 8 GiB in-flight limit and zero draining requests.
Inherited recovery Loop and Token configurations retained only a shutdown-timeout override.
Their resolved Native defaults were 256 MiB in flight and eight draining requests.
The resolved Loop/Token receipt confirms those values.
The prior recovery generator replaces the nested transport dictionary instead of merging it.

These two conditions cannot isolate checkpoint overhead or scheduling policy.
The measured latency values remain valid observations of their respective configurations.
They do not establish the cost of Loop or Token splitting under matched wire settings.
The six-checkpoint control also contains this confound.
Its slowdown must not be attributed only to checkpoint exposure.

New matched-wire configurations explicitly use the Zero values.
Future runs also record the complete resolved Native PDConfig.
Final confirmation must compare common transport limits and draining policy.

## Qualified projection geometry

The active Huginn receipt identifies `down_norm_adapter` with 256-row projection tiles.
The active fix applies to selected MLP down projections and the recurrent adapter.
It does not pad every dense projection to 512 rows.
Any padded-work accounting must use the recorded active fix.
Small Token tiles can add padded work, but that does not by itself establish a Loop-specific architectural benefit.

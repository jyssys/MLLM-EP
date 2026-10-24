# Failures and limits

## Preserved failed attempts

1. `grid_P65408_N1_s0` failed before timing. A corpus fallback indexed a Path object as a mapping.
   The corrected attempt uses `grid_P65408_N1_s0_retry1`.
   The failed source snapshot and traceback remain in the raw archive.
2. `scaling_v1_P2_s0` failed in an added local SplitKV control.
   Pinned FA4 raised `SplitKV not supported on SM 9.0`.
   The corrected campaign retains native S1 and records the unsupported option.
   It does not weaken the registered numerical tolerance.
3. An early campaign attempt encountered the dispatcher lease guard.
   It did not start a competing GPU experiment.
   The campaign now waits for that lease.
4. Nsight Compute returned `ERR_NVGPUCTRPERM`.
   The study did not change driver permissions or request privileged execution.
   Actual HBM traffic, cache hit rate, and occupancy remain unavailable.
5. The first gzip archive exceeded the 95 MiB Git size guard.
   The initial archive remains in the local state directory.
   The delivery uses lossless XZ archives instead.
   All original raw files remain intact.

## Measurement limits

The main native matrix is bounded hardware inference with EOS ignored.
The EOS-enabled probes retain a 128-token cap.
Neither lane establishes unrestricted reasoning accuracy or sustainable SLO capacity.

The operator batch dimension replicates one real model-derived state.
It does not represent independent natural requests.
The main operator samples three physical layers at all four depths.
It does not execute an integrated 96-namespace distributed model.

The initial discovery groups do not contain an at-start experiment-source snapshot.
Later confirmations contain rank-specific snapshots.
The measured attention and merge graph did not change.
Outside-timing preparation and artifact recording did change.

The 1/2/4/8-worker calibration measures warmed steady concurrency.
It does not prove that rolling model startup has zero CPU or I/O interference.
The long-context parallel E2E results varied substantially.
The study adds a dedicated 64K Quiet Lane confirmation instead of selecting the fastest trial.

The local S tuning envelope is an additional diagnostic, not a preregistered online policy.
The S sweep keeps a constant order within each process.
Its large generic local-parallelism effect does not establish a new method.

Only the operator has physical context sharding.
No full distributed model, accuracy-qualified serving candidate, or SLO capacity result exists.
`integration_limits.md` lists the missing KV append and prefill responsibilities.
The native-vLLM Loop PP pause remains unchanged.

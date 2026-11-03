# Profiler-version control for Huginn

Declared on 2026-11-03 after the L2048/O16 eight-second capture failed.

## Observed failure

`short_huginn_s1_nsys384` completes 133 responses before the 60-second no-progress watchdog stops it.
The P worker stack is inside NIXL `transfer`, called by native connector progress.
The capture-start record exists. A normal capture-stop record does not exist.
The matching clean baseline and clean probe run complete.
The old collector is Nsight Systems 2024.6.2.225.

These observations do not prove that the collector version causes the stall.
They justify a controlled collector-version test.

## Single-factor control

Use the Nsight Systems 2025.1.3.140 archive listed in NVIDIA's CUDA 12.9 redistribution manifest.
The archive SHA256 is `dded4227619340307be0ba5bc4e23bcbc966e2df3763170ebb20410c2b54754e`.
Install it in a separate user-owned tool directory.
Do not replace the system profiler, CUDA driver, model runtime, or model weights.
Use a process-local PATH for this controller only.

Sources:

- [NVIDIA CUDA 12.9 release notes](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)
- [NVIDIA CUDA 12.9 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.0.json)

The manifest identifies the bundled profiler version and archive hash.
It does not claim that this version fixes the observed NIXL stall.

Retry the same Huginn L2048/O16 configuration, seed, arrival trace, probe count, capture length, and capture trigger.
Use the existing matched clean run for the first retry.
Retain the old failed run under its original namespace.

The original 5% profile-perturbation limit, frozen Decode SLO, queue-growth screen, service-proximity screen, and exactness checks remain unchanged.
If the first new capture passes all screens, run up to two new independent baseline/probe/capture starts within the existing deadline.
If it fails, preserve the failure. Do not reinterpret an incomplete profile as a successful same-window result.

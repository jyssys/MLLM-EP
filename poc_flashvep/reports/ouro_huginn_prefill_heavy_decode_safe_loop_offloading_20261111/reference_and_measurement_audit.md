# Reference and measurement audit

## Reports read before implementation

The campaign read these complete reports at their pinned Git commits.
It did not substitute later summary text for the original evidence.

| Report | Commit | Retained limitation |
| --- | --- | --- |
| Huginn Loop28 tail diagnosis | 4470f9e2a1c6ea1c1112c59ed751fa921bcaf2dd | Static migration lacked an active execution budget. Resource-level causal percentages remain unidentified. |
| Huginn Output16/4 idle diagnosis | 4470f9e2a1c6ea1c1112c59ed751fa921bcaf2dd | Output4 capture failed sensitivity. Output16 capture failed. Clean idle difference remains unmeasured. |
| Autonomous Prefill compute/headroom | ef3efe173ae48c00925ec616bdb5cc4afddf86a2 | Ouro passes its finite operational screen. Separate D-local probes are not migrated loops. |
| KV admission and profiling recovery | 2d683cce8ebc926c9d410944b705cbcb6f6031a0 | No strict same-run joint success. Full-request credit differs from small-probe credit. |
| Adaptive idle intervention | 0adb7491c866d6d66cf71ea30dca63f82c46d135 | Few actual offloads and long destination waiting. Physical slack remains unmeasured. |
| Native continuation and 2D Oracle | a1c9845ba4f7a56b4e990b16e4f25f96311311de | Native continuation is feasible. The early Token baseline and validated Oracle were incomplete. |

The published GEMM follow-up is commit `30e1717b446e5097a0f523da0bcc890476af3ba1`.
Its Output4 benefit does not qualify the present minimum-Output16 experiment.

## Initial feedback signal

`python -m poc_flashvep.prefill_heavy_decode_safe.gates --audit-prior` returns status 1 on the frozen prior captures.
Output16 has no valid capture. Output4 fails both matched sensitivity comparisons.
Three CPU tests reject tail-only interference, cross-run window composition, and invalid output/context settings.

## Initial collector observations

A fixed-buffer collector removes callback-time allocation and parsing.
It starts before StageEngine builds its graphs. It does not start before the already-loaded model's CUDA context.
It performs no activity toggle or file export during serving.
Synthetic early/late graph self-tests each record all 64,000 expected kernels without invalid or dropped records.
Those self-tests are not model cost or online overhead measurements.

Ouro passes fresh N8/N32 full-state qualification.
Huginn's first N8 combination of CUPTI and asynchronous state observation stops after 2/8 measured outputs.
The owned stalled session stops after stack collection. The incomplete run does not supply performance or capacity evidence.
Its D stack repeatedly reaches scheduling. This differs from the older observed blocking NIXL stack.
The precise cause remains NOT_IDENTIFIED.

The Ouro N64 early capture fails after 37 completed responses with an unspecified CUDA launch failure.
That failure also excludes the run from physical-capacity statistics.

## Library selection hypothesis

The collector compiles with CUDA 12.8 headers, which declare CUPTI API 26.
The failed workers actually load the pip CUPTI 12.9.79 library, which reports API 28.
The recorded process maps establish the loaded library. RUNPATH alone does not establish it.
This version difference is an observed configuration fact, not an established cause.

A new A/B/A control pins the system CUPTI API 26 before Torch imports.
Both marker-free controls and the capture receive that same preload.
Model kernels, weights, workload, and SLO remain unchanged.
The experiment records the preload file hash, actual library maps, and runtime CUPTI API version.

An official CUPTI 12.9.19 archive is available separately if another version control becomes useful.
Its SHA256 matches NVIDIA's redistribution manifest.
The archive and SDK remain outside the publication tree. The evidence retains their provenance only.

## Official documentation

NVIDIA documents that activity-buffer callbacks can delay application submission.
It recommends fast callbacks and preallocated buffers. [CUPTI 12.9 overhead guidance](https://docs.nvidia.com/cupti/12.9/main/main.html).

CUDA 12.x supports minor-version compatibility with limits.
This fact does not establish that a particular profiling configuration works. [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

The CUDA 12.9 toolkit lists a newer packaged driver than this host's driver.
The campaign does not upgrade or replace the system driver. [CUDA 12.9 release notes](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html).

The installed CUPTI header limits hardware-event tracing to Blackwell.
The H100 is Hopper. The campaign does not assume hardware-event tracing is available here.

# Measurement and workload search addendum

This document supplements the current discovery specification.
It does not change the frozen SLO or the five-percent interference screen.
The original evidence and failed configurations remain available.

## Legal memory recovery

The Ouro 16384/16 pilot used two P slots.
Its original D pool could not reserve four complete requests.
The new D pool has 18432 blocks.
The P pool remains at 16384 blocks.

The new marker-free N64 pilot recorded no D-credit refusals in the central observation interval.
Its audited P queue increased by 0.6430630630630634 between interval halves.
The maximum audited queue was seven requests.
Its finite service ratio was 1.1286717181652357.
These are host measurements, not proof of physical P saturation or D capacity.

Huginn 3900/16 failed startup with the 24832-block P pool.
The measured free HBM was 2219311104 bytes.
The required reserve was 3221225472 bytes.
The recovery uses 24064 P blocks and retains the reserve.
The D pool remains at 24832 blocks.

The fresh Native N8 state check passed after this change.
The subsequent N64 pilot still showed frequent D-credit refusals and a decreasing audited P queue.
Refusal prevalence does not establish its causal contribution to throughput.
The 3072-token recovery showed increasing audited P pressure, but its physical and profiling gates remain separate.

## Collector alternatives

The API26 library returned device profiling storage as its actual default.
The returned internal buffer size was 3200000 bytes.
The returned preallocation count was three.
The source records these values before CUDA initialization.

The historical header comment about the default storage location is not authoritative for this loaded library.
NVIDIA records the switch to device storage in its CUDA 12.0 Update 1 release notes.
Memory location remains configurable. [NVIDIA CUPTI release notes](https://docs.nvidia.com/cupti/12.9/release-notes/release-notes.html)

The campaign tested larger preallocated buffers and host-pinned internal storage separately.
Both observer variants passed N8 state checks on both models.
These checks do not establish low profiling overhead.

For Ouro 16384/16, the full-run collector with larger device buffers failed the interference screen.
Its ITL p99 ratios were 1.4155686421359144 and 1.338389926374299 against the two clean controls.
Its maximum-ITL ratios were 1.4056840112333615 and 1.3529815309295996.
The trace remains diagnostic-only.

The five-second collector enabled and disabled actual tracing during Serving.
Its P disable call took 366.1130331456661 ms.
Its D disable call took 5.743669345974922 ms.
Its local ITL p99 ratios were 1.124342039369443 and 1.119679093292699.
The corresponding local maximum-ITL ratios were 1.1495723843785117 and 1.1454652641104768.
This trace also remains diagnostic-only.

Host-pinned internal storage did not pass the screen.
Its local ITL p99 ratios were 1.094970619216496 and 1.1093268068032358.
Its local maximum-ITL ratios were 1.062960322883943 and 1.1145686565989712.
The Native A/A comparison passed for this control set.

Serial kernel tracing failed its synthetic graph check.
It returned 1000 valid records and 63000 invalid records for 64000 expected graph kernels.
It cannot supply a kernel union.
The campaign does not remove invalid records and claim complete coverage.

NVIDIA describes software instrumentation overhead for concurrent kernel tracing.
NVIDIA also distinguishes serial tracing and graph-level tracing.
Graph enclosures do not establish a kernel-interval union. [NVIDIA CUPTI documentation](https://docs.nvidia.com/cupti/12.9/main/main.html)

## Prospective alternatives

The late-activation control defers collector termination until all outputs finish.
It tests whether the measured termination stall explains part of the tail change.
Collection extends beyond the exported interval in this mode.
The unchanged whole-run and local-tail screens still apply.

The quiescent-epoch control traces P throughout the common observation interval.
It traces D only during selected epochs with no active Decode phase.
It disables D collection before any ModelRunner execution.
Actual kernel records determine inactivity inside these epochs.
Queue emptiness does not determine physical inactivity.

A synthetic witness runs after all Serving outputs.
The witness checks collector operation and remains outside performance statistics.
Untraced Decode intervals never become idle in the analysis.
The measured inactive subset provides a lower bound, not a complete idle percentage.
The complete D idle distribution and request-feasible capacity remain NOT_MEASURED unless separate evidence establishes them.
Partial observation does not automatically pass Phase 1.

## Graph-enclosure alternative

The next collector retains actual P kernel records.
It records D graph GPU enclosures and D eager kernels.
It also records copies and memory fills when CUPTI exposes them.
The analysis includes reported transfers from both actor processes.

The union of these D intervals includes graph-internal kernel gaps.
Its complement therefore gives a conservative inactivity bound.
It does not give the exact D kernel busy percentage.
It does not give the complete distribution of physical kernel gaps.
Unobserved RDMA traffic remains a separate limitation.

The synthetic collector check returned 1000 graph records and one eager kernel record.
It returned no invalid or dropped records.
This check does not use model weights or establish low Serving overhead.

The Ouro 18432-token N8 observer passed full state checks.
Its first N64 capture returned one invalid Activity Record at shutdown.
That failed capture remains outside Clean Capacity evidence.
The next diagnostic preserves invalid-record fields instead of discarding them.
It also synchronizes all device streams after Serving outputs.
This shutdown synchronization does not add a measured execution window.

The unchanged five-percent screens still apply to this alternative.
They include Native A/A, whole-run latency, and local Decode tails.
The activity bound alone does not establish safe Loop admission.

## Finite capacity matching

A one-slot P configuration cannot execute a two-request compute batch.
The service screen therefore uses the strongest qualified batch within the configured compute-slot limit.
The analysis also retains the strongest unrestricted batch reference.
The required service ratio remains 85 percent.
This correction does not change a workload, GPU configuration, or SLO.

## Workload search

The Huginn 3900/1024 mixture used real document prefixes and Output16.
Its first N64 pilot had 33 long requests and 31 short requests.
Its audited queue decreased by 2.3763102725366876 between interval halves.
D-credit refusal appeared in 0.8727272727272727 of admission snapshots.
This pilot did not establish the requested compute-ready pressure condition.
Its result is not evidence that every length mixture fails.

The extended Ouro candidate uses 24576 input tokens and Output16.
It remains within the verified 65536-token limit.
It does not change position semantics or model weights.
The initial four-task document search found 142 eligible documents, below the required 152.
The extended search adds other natural LongBench tasks.
It does not use filler, duplicated documents, or synthetic token extension.

Calibration documents and the 128-document confirmation pool remain disjoint.
A short pilot may repeat calibration documents with prefix caching disabled.
Such a pilot does not represent 64 independent texts.
The independent N128 comparison must use 128 distinct documents.

## Evidence and scope

Evidence root: `ouro_huginn_prefill_heavy_decode_safe_loop_offloading_20261111/`.
Relevant plans include `long_native_profile_plan.json`, `short_capture_plan.json`, and `pinned_profile_plan.json`.
Additional plans include `quiescent_capture_plan.json`, `late_capture_plan.json`, and `extended_24k_plan.json`.
Every executed job retains its configuration, source snapshot, model revision, GPU mapping, and failure records.

The current workload search does not establish Loop superiority over Token splitting.
It does not establish exclusive Loop-specific novelty over logical-layer scheduling.
The user requested that separate investigation after this campaign finishes.

## Final measurement alternatives

The graph collector also records matching Runtime API and Driver API intervals.
These CPU intervals never enter the GPU occupied-time union.
An invalid GPU record can leave the observation only with positive evidence that it occurred outside the window.
A pre-window API return alone does not prove GPU completion.
The scope test therefore requires the recorded device fence for that case.

The Runtime-correlation trial located eight missing graph timestamps.
Three corresponding launches started after capture termination.
Five records remained unresolved inside or near the captured interval.
The trial failed its complete-coverage requirement.
The analysis did not replace missing GPU timestamps with CPU API timestamps.

The last alternative uses multi-stream CUDA-event enclosures with D CUPTI disabled in the proposed timing run.
It requires a separate complete-kernel coverage witness first.
That witness returned one invalid D activity record and failed normal closure.
The proposed enclosure-bound capture and its confirmation runs remained NOT_RUN.
The exact D kernel busy percentage also remains NOT_MEASURED for that method.

## Matched graph calibration correction

The Local calibration role did not receive Native P graph geometries.
The original capacity references therefore used eager fallback.
The optional calibration correction supplies those geometries before graph-policy installation.
It does not change Serving or model equations.
The corrected references require actual graph replay, parity, and full recurrent work.
The final host screens retain both original and corrected reference identities.

## Final search limits

Huginn 4080/16 uses exactly the verified 4096-token total context.
Its Native pilot passed Decode limits but did not have growing audited compute-ready pressure.
No physical capacity claim follows from that pilot.

Three real P compute slots also failed to improve matched Native pilots.
The 3500-token TTFT ratio was 3.872020065434701.
The 3072-token TTFT ratio was 1.2279638994819815.
These results use distinct legal memory geometries and remain separate.
The search rejected both slower configurations.

GPU discovery ended without a model passing every Phase 1 gate.
Phase 2 remains NOT_RUN.
The final report distinguishes this measurement limit from a structural lack of all possible Headroom.

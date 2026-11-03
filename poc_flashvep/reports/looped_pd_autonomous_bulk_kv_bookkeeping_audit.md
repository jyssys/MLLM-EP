# Bulk updates to host KV-validity records

Status: GPU correctness tests are in progress. No serving improvement is established by this report.

## Scope

The candidate changes host bookkeeping after a native Prefill CUDA Graph replay.
It does not remove expert work, recurrent work, GPU KV writes, or model operations.
Ouro retains BF16 and R4. Huginn retains BF16 and R32.

The native runtime records each written position for each physical layer and recurrent depth.
The candidate replaces consecutive position updates with an exact interval update.
GPU kernels, operands, output clones, and stream events retain the native execution order.

This is a generic runtime optimization. It does not establish loop-specific novelty.

## Structural accounting

| Model geometry | Prompt rows | Physical layers | Recurrent depth | Position updates per request |
|---|---:|---:|---:|---:|
| Ouro | 4096 | 24 | 4 | 393216 |
| Huginn | 2560 | 8 | 32 | 655360 |

These counts describe host metadata updates. They are not measured GPU operations or latency savings.

## CPU correctness test

The test uses the actual native `_WrittenPositions` class.
It checks existing prefixes, holes, duplicate writes, and consecutive pending positions.

| Test | Result | Evidence class |
|---|---:|---|
| Randomized exact-state comparisons | 20000/20000 passed | MEASURED_CPU_TEST |
| Random seed | 2026110359 | CONFIGURATION |
| Model output parity | Pending GPU tests | NOT_AVAILABLE |
| GPU KV byte identity | Pending GPU tests | NOT_AVAILABLE |
| Request E2E improvement | Not established | NOT_AVAILABLE |

Five CPU timing repetitions used each model geometry.
The timing excludes row grouping, GPU execution, communication, and serving queues.
It must not be reported as an E2E speedup.

Raw results: `looped_pd_autonomous_prefill_compute_decode_headroom/state/bulk_written_cpu_validation.json`.

## Required GPU gates

1. Compare the candidate with the qualified native graph on eight requests per model.
2. Check exact output sequences, transferred state, and full recurrent work accounting.
3. Compare eager, native graph, and bulk-bookkeeping execution on identical 24-request workloads.
4. Measure independent Prefill service before selecting a new pressure level.
5. Run 128-request tests only after correctness passes.
6. Test actual D-local Prefill work against the unchanged Decode SLO.

The controller does not treat increased kernel activity alone as a successful result.
The same-window queue, service, memory, SLO, and probe gates still apply.

## Profiling limitation

Earlier full-interval Huginn captures stopped making response progress inside the native NIXL transfer path.
Those failures remain in the raw evidence.
New bounded captures use separate run names.
Their measured wall-time perturbation must pass the existing acceptance threshold.

The bounded-capture CPU test checks one start, one stop, and unchanged native step calls.
It does not prove that the GPU profiler has negligible overhead.

## Reproduction

Run these commands from the repository root with the qualified environment:

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_autonomous_headroom.test_bulk_written
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_autonomous_headroom.test_timed_capture
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_autonomous_headroom.bulk_written_controls
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_autonomous_headroom.bulk_followup
```

Each GPU run records its source hashes and source archive.
No external runtime file was changed for this candidate.
The final report must include the publication commit and the completed GPU verdicts.

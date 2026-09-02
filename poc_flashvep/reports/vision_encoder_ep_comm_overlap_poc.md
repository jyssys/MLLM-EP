# Vision Encoder ↔ DeepEP Communication Overlap PoC

**Run:** `vision_encoder_ep_comm_overlap_poc_20260902_100000`  
**Date:** 2026-09-02  
**Branch:** `flashvep/vision-encoder-ep-comm-overlap-poc`

## Executive result

`OVERLAP_STATUS: NO_GO` for this bounded real-shape experiment.  A real
Qwen3-VL vision transformer block did not hide either a real DeepEP dispatch
or combine interval.  With paired CUDA-event measurements in one persistent
worker, encoder+dispatch was **12.4% slower** than the encoder and dispatch
serial reference; encoder+combine was **5.0% slower**.  The encoder+expert
negative control was **8.9% slower**, consistent with compute/compute
contention rather than a communication-complementary window.

This closes the measured opportunity for the tested layer-24, M=799 tokens per
EP rank shape.  It does not claim that a future resource-partitioned scheduler
can never help; it says that unpartitioned concurrent streams in this validated
Qwen3-VL/DeepEP path provide no useful headroom and do not justify a full
method implementation.

## Configuration and provenance

| Item | Value |
|---|---|
| Model | Qwen3-VL-30B-A3B-Instruct local BF16 snapshot |
| Model config | 48 decoder layers, hidden 2048, 128 experts, intermediate 768, top-k 8 |
| Runtime | vLLM 0.20.0, PyTorch 2.11.0+cu129, CUDA 12.9, Triton 3.6.0 |
| Parallelism | TP2, DP2 serving workers, EP4, PP1; linear 32 experts/rank |
| Backend | DeepEP high-throughput + `DeepEPHTPrepareAndFinalize` + Triton unquantized experts |
| Mode | BF16, eager, DBO off, prefix cache off, `max_model_len=4096` |
| GPUs | `CUDA_VISIBLE_DEVICES=1,2,3,4` (physical 1–4 only) |
| Image | local `skimage.data.astronaut.png`, one image |
| Real vision unit | one `Qwen3_VisionBlock`, captured input `[1024,1,1152]` BF16 |
| EP shape | exact prior Qwen3 layer-24 capture; 799 tokens/rank after batch-equivalent 4 replay |
| Timing | 10 warmups + 30 measured paired samples per rank and phase; four EP ranks |

Runtime logs prove the mapping `DP0/TP0/EP0`, `DP0/TP1/EP1`,
`DP1/TP0/EP2`, `DP1/TP1/EP3` on physical GPUs 1–4.  The full workload and
capture metadata are in `shape_and_workload_manifest.json`.

## What was audited

The installed vLLM Qwen3-VL path runs `Qwen3_VisionTransformer` blocks from
`qwen3_vl.py`; `encoder_eager_forward` calls the visual module and image input
flows through `_process_image_input`/`embed_multimodal`.  The Qwen3-VL MoE
decoder delegates to `Qwen3MoeDecoderLayer.forward`: attention, post-attention
layernorm, then MoE (`qwen3_moe.py`, lines 416–436).

In `deepep_ht.py`, prepare obtains a previous compute event, computes the
dispatch layout, calls DeepEP `dispatch`, and switches between the compute and
communication streams.  Finalize calls `combine` with the previous event and
waits on the returned event before copying the output.  DeepEP's C++
`get_dispatch_layout`/`dispatch`/`combine` launch on an internal communication
stream and require a different caller stream.  The PoC invokes the Python API
from the worker/default stream, times the internal communication stream, and
waits for returned `EventOverlap` handles.  A prior same-stream assertion was
fixed before the final run; no installed vLLM or DeepEP source was modified.

The candidate is intentionally narrower than prior work.  RESONATOR
([paper](https://doi.org/10.1109/ISCA66397.2026.00173)) studies encoder/LLM
resource sharing and encoder parallelism; no official public RESONATOR source
repository was found in the bounded audit.  SpaceServe
([repository](https://github.com/gofreelee/SpaceServe), checked commit
`66de079af4234b27f7f82ff91d238ef1351324ba`) separates encoder/decoder workers
and uses resource controls.  Flux/COMET
([repository](https://github.com/bytedance/flux), checked commit
`19831ca2d820e3e782ed1d15d8b52d0898b78b26`; [paper](https://arxiv.org/abs/2502.19811))
studies MoE communication/computation overlap.  This PoC does not claim that
overlap itself is novel; it tests the multimodal intersection of a pending
real vision-block unit with current-request DeepEP communication.

## Measurement design

The hook captures a real block input during the first normal Qwen3-VL image
request.  It then pauses at the first real MoE call and, in the same loaded
worker/buffer, replays the exact captured BF16 hidden states, top-k expert IDs,
weights, EP4 mapping, and actual loaded DeepEP buffers.  For each phase,
encoder-only timing is measured on the same encoder stream, followed by
interleaved communication-only and encoder+communication trials.  CUDA events
and rank barriers are outside the work interval; all four ranks complete each
trial before the next collective.  The extra combine after dispatch-only
trials is cleanup and is not included in the dispatch phase.

The `expert` phase is a diagnostic negative control.  It performs the same
real dispatch and Triton expert path, launches the captured encoder block at
the same time, and times the compute phase; combine is cleanup outside the
expert interval.  It is not used to claim a communication result.

## Results

Values below are median across 4 EP ranks × 30 paired samples.  “Serial
reference” is `encoder-alone + phase-alone`; “concurrent” is the actual phase
completion while the encoder block is launched on a separate stream.

| Pair | Encoder alone (ms) | Phase alone (ms) | Serial reference (ms) | Concurrent wall (ms) | Wall reduction | Encoder slowdown | Phase-comm slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Encoder + dispatch | 0.855 | 0.219 | 1.074 | 1.207 | **−12.4%** | +2.4% | +19.0% |
| Encoder + combine | 0.849 | 1.129 | 1.977 | 2.077 | **−5.0%** | +2.1% | +14.0% |
| Encoder + expert (negative control) | 0.843 | 0.968 | 1.811 | 1.972 | **−8.9%** | +3.6% | +4.1% |

The direct DeepEP communication intervals were 0.210 ms dispatch and 0.169
ms combine alone, increasing to 0.250 ms (+19.0%) and 0.192 ms (+14.0%) when
the encoder ran concurrently.  The arithmetic “hidden fraction” is therefore
negative (−63.5% dispatch, −59.2% combine): there is no hidden communication;
the added encoder causes phase contention/serialization.  The p25/p95 values,
CVs, all rank-local samples, and raw CUDA-event intervals are in `summary.json`,
`rank_phase_summary.csv`, and `raw_timings.csv`.

### Stage A: exposed communication headroom

The captured real shape has a measurable communication interval, but it is
small: approximately 0.38 ms for dispatch+combine in the replay.  The normal
one-image live request wall was 4.86–4.89 s in the instrumented process and
3.54–3.60 s in the hook-disabled control.  The latter difference is expected
because the diagnostic intentionally runs 10 warmups + 30 paired replay
iterations inside the first model call; it is reported as instrumentation cost,
not as a production latency claim.  The conservative full-request exposed
fraction is below 0.01% for this bounded invocation, while the per-phase
intervals are precisely measured above.  Thus Stage A is weak at whole-request
scale and does not imply removable latency.

### Correctness and invariants

The original model request completed on all four EP ranks without CUDA,
DeepEP, or NCCL errors.  The hook-enabled and hook-disabled control emitted
the same greedy token ID (`1986`) for both DP requests.  Route IDs, weights,
expert placement, and token order in replay are unchanged (`route_identity:
true`).  Final logits were not exposed by the vLLM `LLM` API used here, so a
logit max-abs/cosine comparison is explicitly **not applicable**, rather than
inferred from token equality.  No encoder output is fed back into the model;
the block is a side-stream diagnostic, so model semantics remain unchanged.

### Resource evidence

`nsys` 2024.6.2 is installed, but a one-request `nsys profile` smoke attempt
failed during vLLM multiprocessing/NCCL startup (`gloo ... Connection closed
by peer`); its log is retained as `nsys_attempt.log`.  `ncu` is not installed.
The valid primary evidence is CUDA-event timing with NVTX ranges
`FLASHVEP_ENCODER_BLOCK`, `FLASHVEP_DEEPEP_DISPATCH`, and
`FLASHVEP_DEEPEP_COMBINE`.  No full serving trace was profiled.

## Gate decisions

| Claim | Verdict | Evidence |
|---|---|---|
| EP communication overhead exists | TRUE (bounded shape) | 0.210 ms dispatch and 0.169 ms combine CUDA intervals |
| Vision encoder and EP communication are resource-complementary | **NO** | Concurrent wall is 5.0–12.4% slower; comm interval itself slows 14–19% |
| Dynamic encoder parallelism adds additional benefit | NOT TESTED | Stage E is correctly gated on overlap GO; no plan switching/scheduler implemented |
| `OVERLAP_STATUS` | **NO_GO** | No positive hiding across dispatch/combine; compute negative control also regresses |

The result is below the 2% “NO_GO” threshold in the required direction: it is
not merely neutral, and there is no stable 5–10% or 10% wall reduction to
classify as HOLD/GO.  The encoder slowdown itself is small (~2–4%), but DeepEP
communication is not free to overlap: the concurrent phase has additional
stream/dependency and communication slowdown.  Therefore Stage E hybrid
encoder DP/TP plans were not run.

## Strongest evidence and counter-evidence

**Strongest negative evidence:** all four EP ranks show the same qualitative
ordering in the final paired run; both communication phases regress, and the
compute/compute control also regresses.  The prior same-stream assertion was
eliminated and not present in the final run.

**Counter-evidence/limitation:** only one local image, one captured layer, and
one real shape (M=799 per rank) were used.  The run is a resource-complementarity
oracle, not a multi-request online scheduler.  A partitioned encoder or SM/HBM
quota could change interference, but that is outside this bounded PoC and is
not evidence for a method today.  The nsys smoke failure also prevents a
kernel-level timeline claim beyond CUDA events.

## Recommendation

Do **not** implement a full cross-request encoder/DeepEP scheduler from this
result.  If the direction is revisited, the single next action should be a
small resource-partitioned validation (one real image block plus one real
dispatch, with an explicit SM quota or process-level isolation) on the same
shape.  Without such isolation, further shape sweeps are unlikely to change
the conclusion that naïve concurrent streams do not expose useful headroom.

## Artifact index

All bounded outputs are under
`poc_flashvep/deepep_revalidation/results/vision_encoder_ep_comm_overlap_poc_20260902_100000/`:

- `shape_and_workload_manifest.json`, `reference_manifest.json`, `source_audit.md`
- `raw_timings.csv`, `rank_phase_summary.csv`, `summary.json`, `gate_summary.json`
- `overlap_rank0.json` … `overlap_rank3.json`, driver output/control files
- `plot1_standalone_timeline.png` … `plot7_encoder_comm_vs_expert.png`
- `negative_control.json`, `nsys_attempt.log`

No external reference repository or model weights were copied into this
repository.

# Resource fallback: 4 → 2 → 1 GPUs

Status: **TESTED_4_2_1 for semantic execution; virtual latency remains unvalidated**

## Four GPUs

The physical TP2/DP2/EP4 path completed 12 deterministic requests and four
bounded official-lane requests. Stock TP4 completed the paired 12-request
reference cohort.

## Two GPUs

The stock TP2 path on physical GPUs 4,5 loaded the same BF16 revision and
completed two OCR image requests. One ended by EOS and one by the deliberate
96-token cap. No local wall time is called EP4 truth.

## One GPU

The normal multiprocess engine stalled during IPC startup and was stopped at a
stage boundary without a partial result. Retrying on physical GPU 4 with
`VLLM_ENABLE_V1_MULTIPROCESSING=0` loaded the full BF16 checkpoint and
completed a 32-token OCR request in 23,905.856 ms. Runtime loading reported
about 58.18 GiB of model memory. This establishes semantic feasibility on one
80GB H100, not serving performance.

## Resume and replay behavior

- Results are appended atomically and completed request IDs are deduplicated.
- Duplicate IDs and missing manifest IDs are unit-tested.
- `--physical-gpus` validates that only 4–7 are requested.
- `--target-ep 4|8` affects only virtual ownership.
- EP4 replay counts are invariant to whether the replay process runs with one,
  two, or four visible GPUs because the target source/owner map is frozen.

The 1/2GPU path is ready for semantic diagnostics and structural virtual replay.
It is not ready to produce validated virtual EP4/8 latency until simulator
calibration passes.

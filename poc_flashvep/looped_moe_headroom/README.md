# Looped MoE expert efficiency

This directory measures three released checkpoints on independent single GPUs.
It does not implement EP, TP, PP, training, or quantization.

The contract is `poc_flashvep/reports/spec.md`.
The original contract remains at its supplied path.
The contract SHA-256 is `91075807b16716b6e186d71062e985e48eec4b7bdb5e33b2a0babfdc52cd333f`.

## Validated environment

The worker Python is `/home/esjung/.venvs/looped-moe-headroom-py312/bin/python`.
The CPU analysis Python is `/home/esjung/anaconda3/bin/python`.
The worker environment uses a read-only package base from the existing Python 3.12 environments.
Its own packages resolve the optional FLA, TileLang, and TVM-FFI import conflicts.
The environment lock records the versions that actually ran.

The native xLLM source remains at `/home/esjung/external/xllm-loop-ep-poc`.
Its pinned commit is `df0ed1ea415b55f0a944a291f90da9d47c98b5d5`.
No tracked native source file changed.
The installation built its official CUDA extension for SM90.
FlashAttention4 uses the official `ENABLE_FLASH_ATTENTION_4` path.

## Entry points

Run these commands from `/home/esjung/MLLM-EP-github`.
Use the saved manifests to repeat their exact configurations.
Completed run IDs are skipped.
Failed or interrupted run IDs require review and a new ID.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.preflight
/home/esjung/.venvs/looped-moe-headroom-py312/bin/python -m poc_flashvep.looped_moe_headroom.prepare --download
bash poc_flashvep/looped_moe_headroom/scripts/check_native_import.sh
/home/esjung/.venvs/looped-moe-headroom-py312/bin/python -m poc_flashvep.looped_moe_headroom.tests.test_contract
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_p0_diagnostic_v5.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_p1.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_calibration.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_heldout.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.followups
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.final_checks
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.analyze
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.plots
```

Do not run dispatchers against occupied GPUs without the resource checks.
The resource helper only stops verified owned burner sessions.
It leaves unrelated processes unchanged.

## Evidence types

`LIVE_MEASURED` denotes an actual GPU measurement.
`LIVE_PROBE_DERIVED` denotes statistics from actual native tensors.
`ANALYTICAL_FLOP_PROPORTIONAL_AMDAHL_NOT_MEASURED` denotes a conditional timing model.
An offline oracle is not a deployable scheduler.
An omitted expert row is not a measured reduction in hardware instructions.

Both official generation paths recompute the complete prefix.
The reported generation TPOT is not cached-decode TPOT.
The short capped rollout tests are not downstream accuracy evaluations.

## Implementation boundary

The adapters load the exact published weights.
The controller temporarily wraps expert calls and router observers.
It restores every hook and function after each intervention.
Omitted rows do not enter the native expert GEMMs.
The grouped kernels can still execute the same number of padded tiles.
The native-weight microbenchmark tests that distinction.

No optimized P3 method is justified by routing similarity alone.
The reports state the actual phase gate and all remaining limits.

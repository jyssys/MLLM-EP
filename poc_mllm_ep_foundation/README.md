# MLLM EP Foundation (Part 1)

This isolated package freezes the Qwen3-VL-30B-A3B-Instruct baseline contract,
replays measured routing onto virtual EP4/EP8 ownership, and refuses to emit
latency for missing or out-of-domain calibration.  It does not implement the
Part 2 resolution/top-k method.

Examples:

```bash
python -m poc_mllm_ep_foundation.cli audit_environment \
  --physical-gpus 4,5,6,7 --output poc_mllm_ep_foundation/artifacts/manifest/environment.json

python -m poc_mllm_ep_foundation.cli virtual_replay \
  --physical-gpus 4 --target-ep 4 \
  --output poc_mllm_ep_foundation/artifacts/traces/virtual_ep4.json
```

`--physical-gpus` controls CUDA resources for semantic/calibration commands.
`--target-ep` controls only the virtual ownership being replayed; it never
turns a one-GPU wall time into physical EP4/EP8 latency.

## Reproducing the completed Part-1 checks

Use the runtime environment only for commands that need vLLM:

```bash
PYTHONPATH=/home/esjung/MLLM-EP-github \
  /home/esjung/.venvs/flashvep-deepep-v020/bin/python3 \
  -m poc_mllm_ep_foundation.cli virtual_replay \
  --physical-gpus 4 --target-ep 4 --waves 0 \
  --output /tmp/virtual_ep4_wave0.json
```

The runner used for measured generation is explicit rather than hidden behind
the manifest-only CLI:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
  /home/esjung/.venvs/flashvep-deepep-v020/bin/python3 \
  poc_mllm_ep_foundation/scripts/run_full_model.py --help
```

Unit tests:

```bash
PYTHONPATH=/home/esjung/MLLM-EP-github \
  /home/esjung/anaconda3/bin/pytest -q poc_mllm_ep_foundation/tests
```

The authoritative completed handoff is
`validation_runs/20260924_completion_v1/frozen_baseline_bundle.json`.
Full MMStar and ChartQA reference/physical-EP4 evaluation, true physical EP2
and EP4 anchors, and the one-/two-GPU semantic acceptance cohort are complete.
The structural EP2/EP4/EP8 replay is functional, but the frozen latency model
remains `SIM_UNVALIDATED` because its untouched EP2 and EP4 error exceeds the
operational gate. EP8 is explicitly an unvalidated sensitivity projection,
not physical EP8 evidence. Exact executed commands and absolute paths are in
`validation_runs/20260924_completion_v1/reports/actual_cli_usage.md`.

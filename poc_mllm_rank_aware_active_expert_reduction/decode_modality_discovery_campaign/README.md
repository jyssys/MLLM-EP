# MLLM decode-modality discovery campaign

This directory implements the bounded campaign specified by
`poc_flashvep/reports/mllm_decode_modality_discovery_campaign_poc_spec.md`.
It starts from commit `7e809abb`, uses only physical GPUs 2,3,6,7, and keeps
discovery and fresh confirmation evidence separate.

The campaign captures one reusable decode trace, evaluates H1--H5 offline,
measures exact functional damage only for frozen policies, and promotes at
most one primary mechanism to confirmation. Raw instrumented runs are local
and append-only; compact evidence, reports, tests, and commands are committed.

## Validated environment

The physical runs use only `CUDA_VISIBLE_DEVICES=2,3,6,7` and the previously
validated DeepEP environment:

```bash
export PY=/home/esjung/.venvs/flashvep-deepep-v020/bin/python
export LD_LIBRARY_PATH=/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD
```

`run_rich_trace_ep4.py` captures the six frozen layers and steps once per
condition. `evaluate_expert_outputs.py` replays the *current checkpoint
weights* for the selected top-8 branches; its output-vector metrics are not a
latency measurement. `run_structured_probes_ep4.py` performs exact drop-only,
globally renormalized interventions and records four teacher-forced logit
steps.  Raw model runs and logs are intentionally ignored; compact JSON/JSONL
evidence is versioned.

The GPU guard is called before and after every physical phase. A representative
guarded invocation is:

```bash
$PY -m poc_mllm_rank_aware_active_expert_reduction.gpu_guard stop \
  --gpus 2,3,6,7 --record artifacts/runtime/stop.json
CUDA_VISIBLE_DEVICES=2,3,6,7 $PY -m \
  poc_mllm_rank_aware_active_expert_reduction.decode_modality_discovery_campaign.scripts.run_structured_probes_ep4 \
  --plan artifacts/manifests/discovery_h1_probe_plan.json \
  --output artifacts/raw_runs/discovery_h1_probes_v5 \
  --control artifacts/control/discovery_h1_probes_v5.json
$PY -m poc_mllm_rank_aware_active_expert_reduction.gpu_guard start \
  --gpus 2,3,6,7 --record artifacts/runtime/restore.json
```

All paths above are relative to this directory in the conceptual example; the
actual module commands are executed from repository root as recorded in the
run manifests and reports.

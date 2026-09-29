# Reproduction and evidence ledger

The authoritative specification is
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/mllm_moe_ep_communication_motivation_poc_spec.md`.
All physical runs used only physical GPUs 2,3,6,7 and the frozen Qwen3-VL revision in
`artifacts/manifests/runtime_audit.json`.

Common environment:

```bash
cd /home/esjung/MLLM-EP-modality-discovery
export CUDA_VISIBLE_DEVICES=2,3,6,7
export LD_LIBRARY_PATH=/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}
export MLLM_COMM_MOTIVATION_ENABLE=1
export PYTHONPATH=$PWD/poc_mllm_ep_communication_motivation/hooks:$PWD
export VLLM_NO_USAGE_STATS=1
PY=/home/esjung/.venvs/flashvep-deepep-v020/bin/python
```

Runtime audit and frozen manifest:

```bash
$PY -m poc_mllm_ep_communication_motivation.scripts.audit_runtime \
  --output poc_mllm_ep_communication_motivation/artifacts/manifests/runtime_audit.json
$PY -m poc_mllm_ep_communication_motivation.scripts.build_manifest \
  --cohort poc_mllm_vision_dominant_captioning/artifacts/manifests/frozen_image_cohort.json \
  --metadata poc_mllm_vision_dominant_captioning/artifacts/ep4_natural_metadata.jsonl.gz \
  --decode-pool poc_mllm_vision_dominant_captioning/artifacts/manifests/frozen_b8_timing_pool.json \
  --output poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json
```

Q1/Q2 and Q3 use versioned output directories; reruns must choose a new suffix because the runners refuse to overwrite existing evidence. Representative commands are:

```bash
export MLLM_COMM_CONTROL=$PWD/poc_mllm_ep_communication_motivation/artifacts/q1_q2_control_NEW.json
export MLLM_COMM_OUTPUT=$PWD/poc_mllm_ep_communication_motivation/artifacts/q1_q2_run_NEW/raw_stage
$PY -m poc_mllm_ep_communication_motivation.scripts.run_q1_q2 \
  --manifest poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json \
  --output poc_mllm_ep_communication_motivation/artifacts/q1_q2_run_NEW \
  --control "$MLLM_COMM_CONTROL" --repeats 3

export MLLM_COMM_CONTROL=$PWD/poc_mllm_ep_communication_motivation/artifacts/q3_control_NEW.json
export MLLM_COMM_OUTPUT=$PWD/poc_mllm_ep_communication_motivation/artifacts/q3_run_NEW/raw_stage
$PY -m poc_mllm_ep_communication_motivation.scripts.run_q3_interference \
  --manifest poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json \
  --output poc_mllm_ep_communication_motivation/artifacts/q3_run_NEW \
  --control "$MLLM_COMM_CONTROL" --decode-tokens 64 --injection-iteration 16 --repeats 3
```

Authoritative local raw evidence is append-only:

- Q1/Q2: `artifacts/q1_q2_run_v7` (`COMPLETE`, 3 repeats per replay condition).
- Q3 clean timing: `artifacts/q3_run_v2` (`COMPLETE`, paired/interleaved 3 repeats).
- Q3 corrected wide trace: `artifacts/q3_run_v3_trace` (`COMPLETE`; actual async mixed iterations retained).
- Same-process overhead audit: `artifacts/q3_paired_overhead_v1` (`COMPLETE`).
- Earlier `v1`–`v6` and no-hook runs are retained locally as non-authoritative debug evidence; none was silently overwritten.

Final compact, pushable evidence is under `artifacts/results/`. It is regenerated with:

```bash
$PY -m poc_mllm_ep_communication_motivation.scripts.analyze \
  --q1-q2 poc_mllm_ep_communication_motivation/artifacts/q1_q2_run_v7 \
  --q3 poc_mllm_ep_communication_motivation/artifacts/q3_run_v2 \
  --q3-trace poc_mllm_ep_communication_motivation/artifacts/q3_run_v3_trace \
  --q3-paired-overhead poc_mllm_ep_communication_motivation/artifacts/q3_paired_overhead_v1 \
  --output poc_mllm_ep_communication_motivation/artifacts/results
```

The full-run and trace-run timing lanes are intentionally separate. Logical byte counts cover BF16 activation payload after token-destination dedup and exclude unmeasured metadata/control bytes. The isolated visual/text physical costs are diagnostic and are never summed to reconstruct the full collective.

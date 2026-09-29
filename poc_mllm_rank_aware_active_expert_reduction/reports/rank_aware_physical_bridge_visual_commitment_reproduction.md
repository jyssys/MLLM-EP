# Reproduction commands

Run from the repository root with the validated environment. Only GPUs
2,3,6,7 may be visible. Stop the recorded burner before a physical run and
restore it in a shell `EXIT` trap.

## Contract tests

```bash
PYTHONPATH=. /home/esjung/.venvs/flashvep-deepep-v020/bin/python - <<'PY'
from poc_mllm_rank_aware_active_expert_reduction.p1_rank_aware_physical_bridge.tests.test_bridge_contract import test_policy_path_and_owner_contracts
from poc_mllm_rank_aware_active_expert_reduction.p2_visual_commitment_event.tests.test_commitment_contract import test_persistent_event_rule
test_policy_path_and_owner_contracts()
test_persistent_event_rule()
PY
```

## PoC-1 physical run and analysis

```bash
export CUDA_VISIBLE_DEVICES=2,3,6,7
export LD_LIBRARY_PATH=/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/poc_mllm_rank_aware_active_expert_reduction/p1_rank_aware_physical_bridge/hooks:$PWD
export MLLM_P1_BRIDGE_ENABLE=1
export MLLM_P1_BRIDGE_CONTROL=$PWD/poc_mllm_rank_aware_active_expert_reduction/p1_rank_aware_physical_bridge/artifacts/control.json
export MLLM_P1_BRIDGE_OUTPUT=$PWD/poc_mllm_rank_aware_active_expert_reduction/p1_rank_aware_physical_bridge/artifacts/reproduction/raw_stage

/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m \
  poc_mllm_rank_aware_active_expert_reduction.p1_rank_aware_physical_bridge.scripts.run_true_ep4_bridge \
  --pool poc_mllm_vision_dominant_captioning/artifacts/manifests/frozen_b8_timing_pool.json \
  --output poc_mllm_rank_aware_active_expert_reduction/p1_rank_aware_physical_bridge/artifacts/reproduction \
  --control "$MLLM_P1_BRIDGE_CONTROL" --repeats 3

PYTHONPATH=. /home/esjung/.venvs/flashvep-deepep-v020/bin/python -m \
  poc_mllm_rank_aware_active_expert_reduction.p1_rank_aware_physical_bridge.scripts.analyze_bridge \
  --run poc_mllm_rank_aware_active_expert_reduction/p1_rank_aware_physical_bridge/artifacts/reproduction \
  --output /tmp/poc1_verdict.json --report /tmp/poc1_summary.md
```

## PoC-2 causal run and analysis

```bash
export CUDA_VISIBLE_DEVICES=2,3,6,7
export LD_LIBRARY_PATH=/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD

/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m \
  poc_mllm_rank_aware_active_expert_reduction.p2_visual_commitment_event.scripts.run_commitment_ep4 \
  --cohort poc_mllm_rank_aware_active_expert_reduction/p2_visual_commitment_event/artifacts/frozen_fresh_cohort.json \
  --output poc_mllm_rank_aware_active_expert_reduction/p2_visual_commitment_event/artifacts/reproduction \
  --raw-output poc_mllm_rank_aware_active_expert_reduction/p2_visual_commitment_event/artifacts/reproduction_raw

PYTHONPATH=. /home/esjung/.venvs/flashvep-deepep-v020/bin/python -m \
  poc_mllm_rank_aware_active_expert_reduction.p2_visual_commitment_event.scripts.analyze_commitment \
  --raw poc_mllm_rank_aware_active_expert_reduction/p2_visual_commitment_event/artifacts/reproduction_raw \
  --cohort poc_mllm_rank_aware_active_expert_reduction/p2_visual_commitment_event/artifacts/frozen_fresh_cohort.json \
  --output /tmp/poc2_verdict.json --report /tmp/poc2_summary.md
```

The PoC-2 analyzer emits `run_skip_stage=false` for this result. Therefore the
spec-mandated K6, rank-aware closure, and cheap-proxy stages must not be run.

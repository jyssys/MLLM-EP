# Actual CLI usage

The verified interface is `python -m poc_mllm_ep_foundation.completion_cli`.
It exposes audit, diagnose_baseline, evaluate_reference,
evaluate_physical_ep, collect_physical_anchor, fit_cost_model,
validate_cost_model, run_virtual, predict_from_trace, resume, export_bundle
and verify_bundle. GPU commands keep per-sample atomic checkpoints and support
`--resume`, `--dry-run-manifest`, and `--raw-log`.

Examples:

```bash
cd /home/esjung/MLLM-EP-github

# Actual one-GPU BF16 semantic generation (resume-safe).
/home/esjung/anaconda3/envs/flashvep-poc/bin/python -m poc_mllm_ep_foundation.completion_cli evaluate_reference --physical-gpus 4 --lane deterministic --manifest /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/data/low_resource_semantic192_manifest.json --output /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/low_resource/reproduce_host1 --max-tokens 128 --max-model-len 10000 --batch-size 1 --max-num-seqs 1 --kv-cache-gib 2 --resume

# Actual two-GPU TP2 BF16 semantic generation (resume-safe).
/home/esjung/anaconda3/envs/flashvep-poc/bin/python -m poc_mllm_ep_foundation.completion_cli evaluate_reference --physical-gpus 4,5 --lane deterministic --manifest /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/data/low_resource_semantic192_manifest.json --output /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/low_resource/reproduce_host2 --max-tokens 128 --max-model-len 10000 --batch-size 1 --max-num-seqs 1 --kv-cache-gib 2 --resume

# Restore the frozen measured EP2 trace, then replay the identical workload as target EP4/EP8.
mkdir -p /dev/shm/mllm_ep_foundation_bundle_trace
tar -xf /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/anchors/ep2_b1_anchor64_interleaved_warm2_measure9_v6.tar -C /dev/shm/mllm_ep_foundation_bundle_trace
/home/esjung/anaconda3/bin/python3 -m poc_mllm_ep_foundation.completion_cli run_virtual --physical-gpus 4 --trace-root /dev/shm/mllm_ep_foundation_bundle_trace --target-ep 4 --output /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/replay/reproduced_ep4.json --cost-model /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/calibration/cost_model_v7_componentwise_frozen.json --prediction-output /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/replay/reproduced_ep4_prediction.json
/home/esjung/anaconda3/bin/python3 -m poc_mllm_ep_foundation.completion_cli run_virtual --physical-gpus 4 --trace-root /dev/shm/mllm_ep_foundation_bundle_trace --target-ep 8 --output /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/replay/reproduced_ep8.json
/home/esjung/anaconda3/bin/python3 -m poc_mllm_ep_foundation.completion_cli project_ep8_sensitivity --replay /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/replay/reproduced_ep8.json --cost-model /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/calibration/cost_model_v7_componentwise_frozen.json --local-kernel /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/calibration/ep8_local_expert16_kernel_v7.json --ep2-to-ep4-transfer /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/calibration/ep2_to_ep4_blind_transfer_v7.json --output /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/replay/reproduced_ep8_sensitivity.json

# Verify every frozen asset by size and SHA-256.
/home/esjung/anaconda3/bin/python3 -m poc_mllm_ep_foundation.completion_cli verify_bundle --bundle /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/frozen_baseline_bundle.json --output /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/bundle_verification_reproduced.json
```

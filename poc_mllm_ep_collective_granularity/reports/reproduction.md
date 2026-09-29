# Reproduction

Run from `/home/esjung/MLLM-EP-modality-discovery`. Only GPUs 2,3,6,7 are authorized.

```bash
export CUDA_VISIBLE_DEVICES=2,3,6,7
export LD_LIBRARY_PATH=/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}
export MLLM_COMM_MOTIVATION_ENABLE=1
export PYTHONPATH=$PWD/poc_mllm_ep_collective_granularity/hooks:$PWD
export VLLM_NO_USAGE_STATS=1

# Ten-repeat clean full baseline. Set MLLM_COMM_CONTROL and
# MLLM_COMM_OUTPUT to new append-only paths before invoking.
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_mllm_ep_collective_granularity.scripts.run_serving \
  --manifest poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json \
  --output poc_mllm_ep_collective_granularity/artifacts/full_clean \
  --control "$MLLM_COMM_CONTROL" --chunk-size 0 \
  --conditions DECODE_ONLY,VISUAL_PREFILL,TOKEN_MATCHED_TEXT \
  --repeats 10 --decode-tokens 64 --injection-progress 16

# Full trace plus CP2048/1024/512/256 clean and trace arms.
bash poc_mllm_ep_collective_granularity/scripts/run_matrix.sh

# Append-only caller/internal iteration alignment traces, including the
# CP256 through-decode capture needed when multiple physical chunks share one
# caller iteration.
bash poc_mllm_ep_collective_granularity/scripts/rerun_aligned_traces.sh

# Exact-hidden/top-k physical MoE-only visual-vs-text replay.
export MLLM_COMM_CONTROL=$PWD/poc_mllm_ep_collective_granularity/artifacts/moe_only_replay_control.json
export MLLM_COMM_OUTPUT=$PWD/poc_mllm_ep_collective_granularity/artifacts/moe_only_replay/raw_stage
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_mllm_ep_collective_granularity.scripts.run_moe_only_replay \
  --manifest poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json \
  --output poc_mllm_ep_collective_granularity/artifacts/moe_only_replay \
  --control "$MLLM_COMM_CONTROL" --repeats 3

python -m poc_mllm_ep_collective_granularity.scripts.analyze \
  --artifacts poc_mllm_ep_collective_granularity/artifacts \
  --output poc_mllm_ep_collective_granularity/artifacts/results

python -m poc_mllm_ep_collective_granularity.scripts.render_reports \
  --results poc_mllm_ep_collective_granularity/artifacts/results \
  --reports poc_mllm_ep_collective_granularity/reports \
  --figures poc_mllm_ep_collective_granularity/figures

PYTHONPATH=$PWD pytest -q poc_mllm_ep_collective_granularity/tests
```

Runtime contract: Qwen3-VL-30B-A3B-Instruct revision `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16, TP2-DP2-EP4-PP1, 32 experts/rank, sparse Triton experts, DeepEP high-throughput, top-k 8, linear placement, eager execution, DBO off, EPLB/redundant experts off, prefix cache off. Chunked arms reserve four tokens/DP iteration for resident decode and give the requested 2048/1024/512/256 remainder to prefill.

The two external-DP replicas use the same four frozen victim prompts so their manually stepped chunked resident phases remain iteration-aligned. These are eight independent requests with cross-DP duplicated content; the injected prefill exists on DP0 only, as in the preceding interference PoC.

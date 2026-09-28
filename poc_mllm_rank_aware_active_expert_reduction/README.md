# Vision-Protected Rank-Aware Active-Expert Reduction PoC

Bounded evidence project for Qwen3-VL-30B-A3B-Instruct. The implementation
keeps Q1 (physical working-set premise), Q2 (visual-protection premise), Q3
(structural upper bound), and Q4/Q5 promotion gates separate. It does not use
MoDES mechanisms internally and does not implement a production runtime.

Only physical GPUs 2,3,6,7 are accepted by the run contracts. EP8 is always
labelled `STRUCTURAL_EP8_PROJECTION_ONLY`.

## Reproduce offline analyses

```bash
cd /home/esjung/MLLM-EP-rank-aware-active
export PYTHONPATH=$PWD

python -m poc_mllm_rank_aware_active_expert_reduction.scripts.analyze_q1 \
  --microbench poc_mllm_rank_aware_active_expert_reduction/artifacts/q1_microbench_gpu2_v4.json \
  --ep2 poc_mllm_rank_aware_active_expert_reduction/artifacts/runs/stage_ep2_v3 \
  --ep4 poc_mllm_rank_aware_active_expert_reduction/artifacts/runs/stage_ep4_v1 \
  --output poc_mllm_rank_aware_active_expert_reduction/artifacts/q1_analysis.json

python -m poc_mllm_rank_aware_active_expert_reduction.scripts.analyze_q3_structural \
  --trace poc_mllm_vision_dominant_captioning/artifacts/ep4_compact.jsonl.gz \
  --output poc_mllm_rank_aware_active_expert_reduction/artifacts/q3_structural_oracle.json \
  --records poc_mllm_rank_aware_active_expert_reduction/artifacts/q3_structural_aggregate.json

python -m poc_mllm_rank_aware_active_expert_reduction.scripts.render_reports \
  --root poc_mllm_rank_aware_active_expert_reduction
```

The physical runners require the frozen DeepEP environment and NVSHMEM library:

```bash
export LD_LIBRARY_PATH=/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_DEEP_GEMM=0 VLLM_NO_USAGE_STATS=1
```

The local `artifacts/runs/` directory is append-only and ignored by Git. Compact
rank counters, joined VDS/damage evidence, analysis outputs, and reports are
committed.

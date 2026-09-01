# Reproduction commands and immutable environment

All GPU runs exported:

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
export FLASHVEP_CAPTURE_EVENT_WAITS=1
export FLASHVEP_FORCE_SYNC_WAIT=0
export WARMUPS=1 ITERATIONS=1 MAX_NUM_BATCHED_TOKENS=8192
./poc_flashvep/asap_sync_phenomenon_reproduction/run_gpu.sh \
  <result-dir> <A|B> <balanced|heterogeneous> <scale> 0 true
```

The calibrated positive control used the same wrapper with
`FLASHVEP_FORCE_SYNC_WAIT=1`, `WARMUPS=1`, `ITERATIONS=1`, and
`DELAY_SWEEP='0 0.5 1 2'` (the exact output is
`..._stageA_H_sweep`).  The DP4/4096 repetition used `WARMUPS=1`,
`ITERATIONS=2` and the same 8192 token budget; the reverse-order pair was
launched as heterogeneous then balanced.  The duplicate `B_B_...rep2`
directory names are retained exactly as produced by that command.

Successful chunk-ablation runs used `max_num_batched_tokens=16384` for both
chunked ON and OFF.  The attempted OFF/8192 run is preserved as a failed
configuration-validation artifact, not included as a timing result.

Model snapshot:
`/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`

Python executable:
`/home/esjung/.venvs/flashvep-deepep-v020/bin/python`

The final aggregate was generated with the system Python only for pandas and
matplotlib (no model execution):

```bash
python poc_flashvep/asap_sync_phenomenon_reproduction/finalize.py \
  --output poc_flashvep/deepep_revalidation/results/asap_sync_phenomenon_reproduction_20260901_final
```

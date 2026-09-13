# Reproduction

All GPU commands below require physical GPUs 4,5,6,7 to be free. The launch
scripts set CUDA_VISIBLE_DEVICES themselves and refuse to start when those
devices contain a compute process.

## Environment

- Python: /home/esjung/.venvs/llada2-flash-sglang-053
- Model: /home/esjung/models/LLaDA2.0-flash-744c3f8
- dInfer worktree: /home/esjung/external/dinfer-llada2-dynamic-block
- Runtime patch: poc_ep_dynamic_block/dinfer_dynamic_block.patch

Apply the patch to dInfer base 6cce0074d7dacf7f91f8cba2df2635761125d344.

## Representative commands

Static clean point:

    poc_ep_dynamic_block/scripts/run_static_point.sh \
      gsm8k 32 32 32 128 0.9 reproduce 1 256

Instrumented shape point:

    poc_ep_dynamic_block/scripts/run_trace_point.sh \
      gsm8k 8 32 8 128 0.9 reproduce 1 256

Actual schedule plan:

    poc_ep_dynamic_block/scripts/run_schedule_plan.sh \
      gsm8k 8 8 128 0.9 256 \
      poc_ep_dynamic_block/plans/exhaustive_128.json 1 8 0

Run the matching evaluate_point.sh or evaluate_schedule_plan.py command after
generation.

## CPU analysis

    python poc_ep_dynamic_block/scripts/aggregate_static.py
    python poc_ep_dynamic_block/scripts/build_static_analysis.py
    python poc_ep_dynamic_block/scripts/analyze_block_traces.py
    python poc_ep_dynamic_block/scripts/build_ep_regime_analysis.py
    python poc_ep_dynamic_block/scripts/analyze_schedules.py
    python poc_ep_dynamic_block/scripts/build_dynamic_analysis.py
    python poc_ep_dynamic_block/scripts/build_compaction_context.py
    python poc_ep_dynamic_block/scripts/compare_fixed_schedule_correctness.py

Clean request timing and observer-heavy stage measurements are intentionally
not pooled.

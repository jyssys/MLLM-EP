# Reproduce the bounded Looped-MoE EP kill-test

Run from the worktree root. Original dirty/staged work remains in the original
checkout; the branch starts at the completed MultiBD evidence commit
`1e96eb6f321bf08bc15560e0f19a14e3c23c59f9`. All new files are scoped to
`poc_looped_moe_ep/`. Do not overwrite the frozen cohort or optimizer policy.

## Pins/environment

- Python: `/home/esjung/.venvs/llada2-flash-sglang-053/bin/python`.
- Torch2.8.0/CUDA12.8, transformers4.57.0, DeepEP1.2.1+73b6ea4.
- Foil HF revision: `c1c857bab4665f00302f191fde5feb274ea97936`, `Foil-1`.
- HF modeling code, not the training checkout, is the actual inference code.
  Remote code is pinned and inspected. The only compatibility alias maps
  public `F.grouped_mm` to available `torch._grouped_mm`, without changing
  the shared environment. All learned tensors remain BF16 and original FP32
  RoPE buffers are restored after HF dtype conversion.
- DeepEP wrapper: unchanged external Diffulex
  `9d1a04e794351f18039b3234a10e8b07e15bcbbf`, with the already validated
  owner-local-ID fix imported from the previous PoC.
- Shape-specific source-SP4 restoration is explicitly timed and never free.
- All commands use physical **4,5,6,7**, logical 0,1,2,3. No other GPU is used.

The source audit also pins the inspected GitHub trees; the HF inference code
adds padding-mask rejection beyond the training-tree version. Do not replace
the HF model with a reinitialized/random model or a manually looped Qwen.

```bash
export PYTHONPATH="$PWD:/home/esjung/external/Diffulex-multibd-poc"
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.audit_checkpoint
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("ShourenWSR/how-to-loop-moe", revision="c1c857bab4665f00302f191fde5feb274ea97936", allow_patterns=["Foil-1/*","modeling_loop_lm.py","loop_trace.py","README.md","LICENSE","NOTICE"])'
```

The existing freeze is WikiText-2 raw test, dataset revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, metadata-only seed3407,
8 calibration +24 held-out paragraphs. Prefix64/128, teacher-forced offsets
0/4/8, natural greedy4 for correctness. These are short base-model probes,
not a reasoning-accuracy benchmark or a fresh Wikipedia-article-disjoint set.
The original freeze command was `python -m poc_looped_moe_ep.scripts.freeze_requests`;
it refuses to replace an existing frozen manifest.

## Scoped physical commands

Use NEW output directory names on rerun. The guard stops/restores only the
exact same-user 4/5/6/7 burner and refuses foreign target-GPU competition.
It reuses the existing watchdog/cleanup implementation without changing prior
files. The compatibility ownership marker is named `POC_MULTIBD_OUTPUT` but
points inside THIS PoC, so scoped orphan cleanup cannot target another run.
On completion, the existing burner argv/environment is restored, and its
append-only stdout is archived outside the worktree with a stable snapshot
and archive locator retained in the run directory. The launcher default is
2/3/6/7, so manual restoration MUST override `VLLM_UTILIZE_GPUS=4,5,6,7`.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.run_guarded --output results/raw/repro_gate --timeout 600 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.validate_true_ep4 --output "$PWD/poc_looped_moe_ep/results/raw/repro_gate"
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.run_guarded --output results/raw/repro_cal --timeout 1800 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.run_physical_static --output "$PWD/poc_looped_moe_ep/results/raw/repro_cal" --split calibration --placement S0
```

The original calibration directory was `results/raw/calibration_S0`.
`optimize_best_static` consumes only that calibration data and refuses to
overwrite `results/frozen_best_static.json`: 14 starts ×8000 bounded swaps,
16 experts/rank, minimize invocation-wise max assignments; deduplicated remote
destination rows break ties. It is not a certified physical-latency solver.

The completed held-out command was:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.run_guarded --output results/raw/held_out_paired --timeout 2400 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.run_physical_static --output "$PWD/poc_looped_moe_ep/results/raw/held_out_paired" --split held_out --placement both
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.run_guarded --output results/raw/copy_probe --timeout 300 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.measure_copy --output "$PWD/poc_looped_moe_ep/results/raw/copy_probe"
```

Reference computation runs BEFORE expert sharding. Only CPU reference results
remain; no full reference expert weights stay resident on GPUs during EP
timing. S0/S1 setup copies are outside warmup/timing and occur between whole
episodes, never within loops. That exclusion is appropriate for two STATIC
arms, not permission to ignore a hypothetical dynamic migration bill.

Two warmup episodes, three clean and three event-only repeats, one heavy trace.
An episode is three native full-prefix forwards. S0/S1 order is reversed on
alternating requests; clean/event order is also counterbalanced. Event results
are deferred. No per-loop host global sync is added by the observer. Dispatch
and Combine are API/layout/coordination intervals, not pure wire bandwidth.

## Offline reproduction (no GPUs; keep burners running)

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.analyze_loop_conflict
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.validate_artifacts
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.analyze_migration_roi
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_looped_moe_ep.scripts.make_plots
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m unittest poc_looped_moe_ep.tests.test_contract
```

`python -m poc_looped_moe_ep.scripts.final_audit` separately checks this
workspace's original tracked-diff hashes, the selected GPU-resident engine
processes' burner ancestry, unit tests and raw-file checksums. It records
`completion_audit.json`; the original dirty-state hashes are specific to this
handoff, not an invariant that another checkout must recreate.

Failed `physical_gate_v1` is preserved: native-layout `zeros_like` inherited a
noncontiguous stride, which NCCL all-reduce rejected. `physical_gate_v2_contiguous`
passes after a contiguous publication buffer; all four native/EP logits match
bitwise. The regression test calls the actual allocation seam. An initial
environment record with an extra-newline spec snapshot is separately preserved;
the final snapshot/environment SHA matches the original exactly.

Use JSON/request-level CIs for interpretation. Layers, loops, ranks, repeats
and overlapping prefix positions are not independent statistical samples.
S2 candidate/ideal bars are load accounting only; there is NO latency oracle
or executable dynamic migration method hidden in the reproduction.

Deliverable consolidation: `validate_true_ep4.py` and `runtime.py` perform
the loop-sharing/identity checks; `run_physical_static.py` captures routing;
`analyze_loop_conflict.py` computes the structural candidate and ideal bound.
Separate `verify_loop_semantics.py`, `trace_routing.py` and
`estimate_oracle_bound.py` wrappers are therefore not needed. A validated
timing-oracle model, nonlooped physical control and Stage C serving benchmark
are NOT_RUN after the representativeness/materiality gates; no empty scripts
or invented measurements stand in for them.

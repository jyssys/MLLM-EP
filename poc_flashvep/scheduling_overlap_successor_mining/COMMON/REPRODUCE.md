# Reproduction and evidence boundaries

This directory belongs to **Layered Prefill / FastPP / NanoFlow**. It is not the
SERE/Libra/MoDES study. Start from `ARTIFACT_INDEX.md`, the main report and the
three `PAPER_AUDIT.md` / `CODE_AUDIT.md` documents.

## Hardware and source state

Only physical **4,5,6,7** are authorized. Native Layered TP2 uses 4/5; do not run
burn on the other pair while any experiment is measuring. `run_scoped.sh` masks
the four authorized devices; individual launchers/guards verify their scope.
Run GPU engines serially. Do not terminate unrelated jobs to reproduce a table.

`EVIDENCE_BUNDLE.json` enumerates selected small Git evidence with SHA256. Full
raw data is retained locally under:

`poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/`

Reference source pins, environment distribution versions, actual Python module
locations and compatibility-patch hashes are in
`cpu_analysis/environment_provenance.json`. Large checkpoints, cloned sources,
compiled environments and Nsight binaries are intentionally not vendored.
Use each native environment and documented patch, not one shared vLLM install.

## CPU analysis commands

From repository root; set `study` and `result` to the two explicit paths below.
These task-specific shell variables do not change any system environment path.

```bash
study=poc_flashvep/scheduling_overlap_successor_mining
result=poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510
bash "$study/COMMON/run_cpu_only.sh" /home/esjung/.venvs/scheduling-fastpp-py310/bin/python \
  "$study/COMMON/build_interim_tables.py" --results "$result"
bash "$study/COMMON/run_cpu_only.sh" /home/esjung/.venvs/scheduling-fastpp-py310/bin/python \
  "$study/FASTPP/analyze_restart_screen.py" "$result/fastpp_runs/qwen3_full_warmup_chunks_20260909_v1"
bash "$study/COMMON/run_cpu_only.sh" /home/esjung/.venvs/scheduling-fastpp-py310/bin/python \
  "$study/FASTPP/analyze_policy_envelope.py" "$result/fastpp_runs/qwen3_full_warmup_chunks_20260909_v1" \
  --scope native_Qwen3_PP4_full_workload_warmup_existing_five_knobs_only
bash "$study/COMMON/run_cpu_only.sh" /home/esjung/.venvs/scheduling-fastpp-py310/bin/python \
  "$study/COMMON/analyze_slo_envelope.py" "$result/fastpp_runs/qwen3_full_warmup_chunks_20260909_v1"
bash "$study/COMMON/run_cpu_only.sh" /home/esjung/.venvs/scheduling-nanoflow-py310/bin/python \
  "$study/NANOFLOW/analyze_cohort_screen.py" "$result/nanoflow_runs/decode_volume_portfolio_20260909_v1"
bash "$study/COMMON/run_cpu_only.sh" /home/esjung/.venvs/scheduling-nanoflow-py310/bin/python \
  "$study/NANOFLOW/analyze_plan_envelope.py" "$result/nanoflow_runs/decode_volume_portfolio_20260909_v1"
```

Each run manifest records the actual GPU command, request trace, randomized
order, start/end and completion state. Reuse those with a **new result name**;
launchers refuse to overwrite existing screens. The manifests, rather than an
invented universal command, are the exact per-experiment run recipes.

## Interpretation safeguards

- Request rows are not independent engine restarts. Bootstrap units are restart
  pairs; small-n intervals are descriptive.
- Fixed-cohort NanoFlow is not its full asynchronous online scheduler. Manual
  native FFN plans are not its paper-searched optimum. Initial capture remains
  in request E2E. Steady ITL and setup-inclusive E2E answer different questions.
- NanoFlow unequal-output paired results remain excluded from performance
  acceptance. Near-tie HF diagnostics are not a broad benchmark certification.
- LP/FastPP native MoE topology and the common Qwen3-VL DeepEP transfer path are
  different experiments. Do not call their combination a native VL/PP×EP port.
- Finite existing-option lower envelopes do not bound every legal execution.
  Stage maxima, resident communication overlap and zero-TTFT bounds are never
  credited as measured scheduling/removable E2E gains.
- The GPU ledger unions intervals per physical GPU. It is conservative recorded
  experiment time, not device-busy time; loading/build/JIT/burn are excluded.
- No large successor or Kimi run is justified without a material, non-trivial
  Qwen request-level gate. Missing native semantics must stay explicit.

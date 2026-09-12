# Refinement-Aware Dynamic Wave Sizing PoC

This workspace tests whether LLaDA2.0-Flash refinement state changes the
physical sparse-EP shape enough to justify changing model-forward wave size.

The immutable comparison is best-static true EP4 versus a per-step oracle.
All GPU launches are restricted to physical GPUs 0,1,2,3.

Final result: **NO-GO**. Best static is mini32 (6.273 s median BCT); the
ready-set-feasible zero-cost dynamic oracle improves it by only 0.650%.

Reproduce the CPU analysis from the retained raw traces with:

```bash
python poc_refinement_wave_sizing/scripts/analyze_static_sweep.py \
  poc_refinement_wave_sizing
python poc_refinement_wave_sizing/scripts/evaluate_quality.py
python poc_refinement_wave_sizing/scripts/analyze_wave_traces.py \
  poc_refinement_wave_sizing
python poc_refinement_wave_sizing/tests/check_analysis_invariants.py
```

The external dInfer changes are preserved in
`patches/dinfer_true_ep4_and_raws_trace.patch`. Raw logs and per-rank JSONL
traces remain local and are intentionally excluded from Git; collapsed CSVs,
figures, clean generated outputs, and all analysis code are versioned.

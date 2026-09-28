# Q2-3 visual-grounding budget gate

This bounded PoC asks whether a single-pass, request-specific visual grounding
signal at `(request, decode step, layer)` predicts exact structured K8-to-K6/K4
functional damage. It does not select experts, implement the later rank-aware
policy, compare against MoDES, or claim latency gains.

The frozen protocol is in `research_spec.md`. Raw instrumented runs are local
and append-only; frozen manifests, compact evidence, reports, tests, and the
command ledger are committed.

## Result

The frozen Phase-1 gate failed on 24 fresh held-out requests. The
vision/text-separated grounding score neither predicted structured K8-to-K6/K4
damage nor its same-token image-swap delta, and it did not improve over the
calibration-selected full-prefill control. A post-gate distributional/JSD
diagnostic also found no visual-specific rescue. Consequently Phase 2, runtime
expert selection, and latency measurements were not run.

See `reports/final_summary.md` for the compact verdict and
`reports/nonlinear_distributional_diagnostic.md` for the exploratory KL/JSD-
style analysis.

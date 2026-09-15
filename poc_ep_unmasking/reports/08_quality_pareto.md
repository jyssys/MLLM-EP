# 08 — Quality and latency Pareto boundary

Only stock O0 quality was measured: GSM8K 5/8 exact answer and HumanEval 2/8
functional success on the bounded eight-request samples. The scores are
coarse and these short generations are not a benchmark-wide accuracy claim.
GSM8K clean and observer-heavy final answers/generated lengths were identical.

No alternate unmask trajectory was executed, so zero-quality-drop E2E gain,
small-epsilon Pareto gain, NFE drift, and final-sequence divergence are
unknown. Delta 0.05/0.1 CPU reanalysis shows that larger confidence windows
produce more candidates, but these windows were **not** validated as
semantically safe. Even then the frozen-route one-swap rank-load reduction
averaged only 0.63% and 1.34% combined, respectively, before any physical
latency conversion.

`QUALITY_PARETO.csv` contains stock measured points and an explicit not-run
marker for alternatives. It does not plot an invented speedup/quality curve.

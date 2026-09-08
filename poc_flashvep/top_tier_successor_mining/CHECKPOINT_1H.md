# One-hour checkpoint — 2026-09-07 15:00 KST

The three papers and public implementations/availability have been audited before
serious runtime ports. No winner selected, no successor implemented.

## Baseline milestone status

| Milestone | SERE | Libra | MoDES |
|---|---|---|---|
| Paper | Main/appendix read | Final proceedings + algorithms read | CVPR main/appendix read |
| Code | Official kernel compiled, exact route parity | Official URL unavailable; explicit paper port | Official scoring/ablation parity |
| Sanity | No-op and native baseline quality checks | 120 planning invariants; actual next-gate prediction | No-op exact, official/paper ablation differentiated |
| MLLM probe | GQA/ChartQA fresh paired greedy | 64 real-image requests, all layers | 64 pilot; full 1,024 calibration active |
| Material failure | ChartQA pilot -17.2pp at S2/.5 | Not established | Not established |
| E2E headroom | Pending real EP port | Pending cost/overlap validation | Pending selected thresholds + fast skipping |
| Prior-art screen | Rerouting/calibration literature pending exact candidate | PROBE adjacency already identified | AnyExperts/MACS adjacency identified |

## What might kill the emerging signal

SERE's ChartQA loss is NOT yet a successor candidate. S4 or rho=.7 recovered the
pilot quality, and decode-only application is an even cheaper attack currently
running. Appendix C.2 already says prefill has no expected speedup. If restricting
application to decode fixes quality while preserving useful speed, this is
engineering scope selection, not a successor paper.

Libra's vision recall 79.1% versus text 85.5% does not imply failure. The published
range is already broad, and misprediction must be translated into lost useful
replication/overlap and direct request latency before promotion.

MoDES's input-versus-zero ablation changes layer rankings (pilot Spearman .48 text,
.70 vision), but fixing an implementation discrepancy alone is not a paper-core
problem. Both semantics will be evaluated, with official behavior as baseline.

## Next

1. Complete full official MoDES calibration and unchanged frontier search.
2. Finish SERE phase/trivial-fix controls, larger held-out quality sample.
3. Verify actual DeepEP HT EP4 route/skip semantics and request-level timestamps.
4. Libra predicted vs exact replication plans, measured real-weight transfer and
   local/remote computation windows; retain original-system reproduction limits.
5. Rank only after the same seven milestone fields are honestly populated.

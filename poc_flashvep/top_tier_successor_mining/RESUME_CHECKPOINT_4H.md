# Four-hour resumption checkpoint — 2026-09-08 15:10 KST

Resumption began approximately11:03 KST. Only physical4/5/6/7 are used;
burn is stopped while the experiment queue is resident. Other users'0–3 jobs
are not touched.

## Evidence now available

- All eight native Libra VL groups finished:32 distinct real requests,ten
  randomized repetitions/group. A perfect-current-route prediction control
  improves paired prefill by only0.146% median across groups; group-bootstrap
  95%CI[-0.102,1.408]%. This is a bounded native execution diagnostic,not full
  request E2E or a universal scheduling upper bound.
- Native vanilla and Libra agree on all320 first-token comparisons. Both
  disagree with the HF capture for one of32 requests. Preserve this approximate
  bridge limitation; do not label it a Libra-specific quality failure.
- SERE's strongest B1 quality loss is largely removed by existing knobs:
  S4 recovers100%/88.9% of the observed ChartQA/GQA losses. Original settings,
  HF controls and natural EP measurements all exist. No new mechanism is needed
  for that recovery in these samples.
- Corrected MoDES mask-lifetime screen is collecting the last two of six
  conditions. The old per-layer metadata cost is explicitly a port defect.
- Fixed-length and algorithmic no-op controls show a substantial port-cost floor
  without intended model changes. Through-first-EOS parity is100%; forced
  post-EOS trajectories are not universally bitwise identical.

## Remaining bounded checks

1. Finish all six corrected MoDES conditions and refresh confidence intervals.
2. Native Libra M8192/source tests whether the observed short-window limitation
   persists at32K aggregate tokens. Do not treat shape variants as new ideas.
3. Nsight at M2048/source records actual native overlap/CPU-launch behavior.
   Profiled latencies are not clean speed measurements.
4. Only after these checks, rank all three. Kimi and a successor prototype are
   conditional on a material nontrivial survivor, not required busywork.

Accounting is maintained in `resume_20260908/GPU_TIME_LOG.summary.json` under
the result root. Its GPU-resident process intervals include loads and CPU control;
they are not CUDA-active time. Burn contributes zero research hours.

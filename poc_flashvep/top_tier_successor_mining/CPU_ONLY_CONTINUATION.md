# CPU-only continuation / GPU release checkpoint

User resource change: 2026-09-07 17:37 KST. GPU1–4 released, no burn, no automatic
restart. Original research contract remains applicable to evidence standards;
its remaining live-GPU milestones are explicitly deferred, not silently waived.

## What can still be done without GPU

1. Complete original-paper/supplement implementation audits and prior-art attacks.
2. Re-score held-out predictions, clustered uncertainty and failure distributions.
3. Analyze captured real routes using the supplied Libra CPU planner, keeping
   planner statistics separate from actual distributed runtime measurements.
4. Build honest quality Pareto tables and conditional bounds from existing data.
5. Separate simple fixes, known behavior, port limitations and genuinely unresolved
   candidate causes. Specify the cheapest decisive future GPU experiments.

## What cannot be claimed without the deferred experiments

- Quality-matched real-serving successor speedup for any of the three methods.
- Full-layer/full-VL correctness and request performance of the Libra bridge.
- Held-out quality of the newly calibrated full MoDES frontier.
- New calibrated SERE quality/E2E or Qwen+Kimi generality.
- A validated all-three final ranking or FOUND_SUCCESSOR_STRONG_GO.

## Preserved state

Result root: `../deepep_revalidation/results/top_tier_successor_mining_20260907_135950/`.
MoDES importance: `analysis/modes_alpha1024_official.json`.
Completed first frontier target: `quality/modes_frontier1024_grid100/frontier.json`.
Partial reusable search cache: same directory `evaluations.jsonl` (301 unique points).
SERE confirmatory similarity: `quality/sere_fineweb_400x128_official_norm/similarity.pt`.
Libra capture manifest: `quality/libra_vl_bridge_inputs/workload_manifest.json`;
GPU tensors were not captured. Six CPU bridge boundary checks passed.

## Future resume, only when user reauthorizes GPU availability

1. Revalidate physical mapping/free memory; do not touch GPU0/5/6/7.
2. Explicitly update GPU_EXECUTION_POLICY.json; never bypass it implicitly.
3. Resume `modes_frontier.py` with identical saved arguments/output directory;
   it reconstructs the teacher cache and validates one repeated cached point.
   Use a NEW raw log filename to retain the interrupted log.
4. Complete full Libra parity and clean EP request comparisons before ranking.

The stopped `resume_baseline_screen.py` process is not a daemon and is not running.

# Decode-first dynamic-depth-aware DP discovery

The Stage-0 primary result is `QUALITY_FAIL` for the registered adaptive configuration.
This result does not establish zero DP scheduling headroom.
The study does not proceed to a live DP4 placement baseline, simulator, depth oracle, or predictor.

Start with [the final report](reports/final_report.md).
Use [the reproduction commands](reports/reproduce.md) to inspect or repeat the evidence.
[The timing contract](reports/timing_contract.md) separates measured work from unmeasured scheduling speedup.

The primary checkpoint is `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The tested policy uses q=0.5, minimum R2, maximum R4, and `last_exited` KV propagation.
Only physical GPUs 4, 5, 6, and 7 execute model work.
The source specification remains unchanged in `reports/spec.md`.

`artifacts/quality_pilot_analysis_v2` contains the 32-question-per-domain primary quality comparison.
`artifacts/pilot_export_v2` contains the actual online depth trace, CSV tables, and figures.
The earlier pilot, failed runtime attempts, and per-run source snapshots remain available.
The longer-cap sensitivity remains separate from the common-cap primary analysis.

Local numerical arrays remain outside Git under the task's output directory.
Their file sizes and hashes appear in `artifacts/local_array_manifest.json`.
The recorded physical commands regenerate those arrays.
No checkpoint weights enter this task's commit.

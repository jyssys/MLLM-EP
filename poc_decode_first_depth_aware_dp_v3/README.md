# Decode-first DP discovery, v3 continuation

The final verdict is `NO_GO_SCHEDULING` for this pilot.
This directory preserves the v2 evidence and continues the experiment under the v3 specification.
The v2 parent commit is `e6a350e0a862d94251c5cefa0ef38038b017b8c4`.
The selected runtime remains the CDB authors' Base Ouro-1.4B implementation.
The previous quality failure remains a warning. It does not stop scheduling discovery under v3.

The allowed physical GPUs are 4, 5, 6, and 7.
Each runtime process sees its assigned physical GPU as logical CUDA device 0.
This is independent DP4, not tensor parallelism.

Reports separate physical measurements, calibrated models, and clairvoyant oracle estimates.
Read `reports/final_report.md` for the final gate results.
Read `reports/timing_contract.md` for metric definitions and limitations.
Read `reports/reproduce.md` for commands.

The publication stores large per-rank traces as lossless gzip files.
Their uncompressed originals remain on the server.
The new JSON reader accepts either form.
No previous artifact or dirty file is removed.

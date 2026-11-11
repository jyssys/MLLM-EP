# Huginn Loop28 and Output16/4 diagnostic evidence

This directory supports two reports:

- [Loop28 Decode-tail diagnosis](../huginn_loop28_decode_tail_diagnosis_final_summary.md).
- [Output16/4 idle-capacity diagnosis](../huginn_output16_output4_physical_idle_diagnosis_final_summary.md).

The reports retain their original measured values and limitations.
This publication does not add GPU experiments or change the frozen SLOs.

## Published files

`configs/` contains the diagnostic configurations.
`analysis/` contains the readable summaries and request-level CSV files.
`figures/` contains the figures cited in the reports.
`logs/` includes failed profiling attempts.
The diagnostic source is in [`poc_flashvep/huginn_loop28_tail_diagnosis`](../../huginn_loop28_tail_diagnosis/).

## Raw evidence

`evidence_bundle/` stores lossless archives of the original `raw/` directory.
Its manifest records each file's byte count, SHA256, and ordered chunks.
The existing helper `poc_flashvep.completion_aware_loop_pd.evidence_bundle.verify` checks and extracts the archives.
Pass the bundle directory as `target` and a new empty-path destination as `destination`.
Extracted paths are relative to the original `raw/` directory.

The publication excludes Python bytecode and lock files.
It also excludes the live Burner log, compiled libraries, and the private external-runtime patch.
The external runtime remains unchanged locally.
Build commands and source hashes remain available in `tools/`.

Large derived event timelines and kernel-interval JSON files remain local.
They duplicate the archived raw timing records and can be regenerated with the published diagnostic source.
The smaller analysis JSON files cited by both reports are directly available here.

## Preservation

The publication uses a separate Git index.
It does not change the active checkout, existing staged entries, or external runtime diff.
No GPU experiment starts during publication. Approved burners remain active.

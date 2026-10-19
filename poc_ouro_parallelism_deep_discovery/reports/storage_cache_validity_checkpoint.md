# Compilation-cache validity checkpoint

## Status

The v1 storage-control arms used shared native compilation-cache directories.
The copies arm loaded compiled artifacts from the preceding alias arm.
This is a confirmed cache-reuse confound.
It does not prove the cause of the observed numerical differences.

Parameter assertions confirm distinct allocated weights with equal BF16 words.
Those assertions do not establish which parameter aliases a loaded compiled artifact consumes.
The v1 timings cannot establish a physical-sharing or HBM-residency effect.
The v1 raw evidence remains unchanged.

The coordinator stopped only the owned P4 queue coordinator.
The already active guarded P4 alias job continues to completion.
No foreign process or predecessor experiment changed.
The queue state file can retain its last running state after that interruption.
This report records the interruption instead of rewriting that raw state.

## Corrected control

The new control uses an explicit, fresh `compilation_config.cache_dir` inside each job directory.
The analyzer permits only that expected cache-path difference between engine configurations.
Both arms still allocate all four equal-valued block copies before timing.
The new runs have the suffix `isolated_v2`.

Fresh cache isolation does not itself prove numerical equivalence.
The analysis must compare logits, hidden state, sampled KV, and compiled execution evidence again.
Natural-generation accuracy remains separate from forced-output hardware timing.
No tolerance changes after a failed comparison.

## Evidence

- Initial results: `artifacts/storage_control_checkpoint1.json`.
- Initial cache paths: the corresponding v1 `console.log` files.
- Corrected queue: `scripts/run_isolated_storage_controls.py`.
- Cache option: `scripts/native_baseline.py`.
- Source audit: `reports/storage_numerical_audit.md`, when complete.

No cache-hit-rate, HBM-traffic, candidate speedup, or natural-quality claim follows from this checkpoint.

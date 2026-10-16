# Original reference harness recovery

The initial HF reference run predates the source ZIP feature.
Its run contract records the original harness SHA256.
The current file also contains later filename handling and supplemental-input options.
The recovery removes those known additions in memory.
The recovered bytes match the original recorded SHA256 exactly.

The original teacher and natural model calls both specify `exit_at_step=3`.
The original R4 reference did not rely on the default threshold selector.
The recovered source remains separate from the current executable harness.
No old raw result, current model source, or primary generation changes.

This check resolves a harness-source provenance gap.
It does not resolve the failed state-space criterion or reproduce the private author evaluator.
The later threshold probe remains a hypothetical-selector diagnostic only.

Evidence: `results/HF_original_source_recovery.json` and `evidence/recovered_sources/run_hf_fidelity_original.py`.

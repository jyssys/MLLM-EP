# Executed verification and limitations

CPU tests: `python -m unittest discover -s poc_dispatch_combine_asymmetry/tests -v` — **8/8 PASS**. Covers actual geometry mismatch, invalid IDs/source, destination dedup, empty/dummy paths, idle forward exclusion, incomplete/duplicate layer/rank rejection, one-rank component consistency and repeat-level bootstrap unit.

CPU independent published-bundle reanalysis: `scripts.verify_bundle` — **PASS**. SHA-verified all compact raw records; exact same metrics/CIs from 90 final clean waves, 9 arm-size strata ×10 repeats. Reconstructed all432 layer traffic accounts from actual route IDs/weights, checked all1,728 runtime owner maps, validated five kernel classes ×48 calls in all12 profiled native rank/arm ranges. Known invalid first attempt remains rejected. Exact verification counts are in `artifacts/bundle_verification.json`.

Compilation: `python -m compileall -q poc_dispatch_combine_asymmetry` — PASS. All five generated plots checked; the internal-composite plot is explicitly UNVALIDATED/perturbed.

Actual GPU evidence is NOT replaced by these tests: clean native VL/LLM EP4 runs, bounded route-controlled physical DeepEP replay and separate Nsight profiling were executed successfully; run manifests/status and raw timing records are published. No silent backend fallback: actual prepare class must be DeepEPHTPrepareAndFinalize; loaded expert count=32/rank and top-k=8 asserted, all expert maps audited. Physical GPU↔logical EP ranks are recorded independently of PID/CUPTI device IDs.

Failures preserved, not cherry-picked away:

- Initial high visual collection mixed32 external-DP idle forwards into a common wave label. Strict analyzer rejected incorrect layer counts. Fixed the CPU DP-token selector; new clean runs are the final evidence. Old raw attempt is retained/excluded.
- First replay launcher used a nonexistent `torchrun` executable. It did not execute a measured replay. Corrected `python -m torch.distributed.run` completed with finite watchdog.
- Internal API/capture instrumentation perturbed some components by >10%; those durations are UNVALIDATED diagnostic, not headline baseline. Source-backed kernel classes still describe fused boundaries, not pure transport time.

Unmet/limited claims explicitly retained:

- Standalone LLM measured in a separate loading block, not10 paired model-restart blocks. Same-VL visual/text primary comparison IS interleaved/paired10. Cross-model timing is descriptive only.
- One frozen input per size; repeat CI does not prove image-population generalization. Different size images are not a same-image resolution causal sweep.
- Common active-TP0 model envelope overhead is8.34% visual/7.65% text, not a zero-overhead claim. It includes instrumentation drain cost and excludes vision encoder; standalone LLM has only n5 host-wave bypass evidence.
- Pure payload movement vs packing/polling and exclusive tail/control fractions cannot be separated on the installed fused backend. EffectiveBW=N/A. Profiler durations are not substituted into clean latency.
- Actual producer-ready replay covers only layers0/24/47. It removes several non-transport runtime factors together, so producer-readiness attribution is INFERRED, not an exclusive encoder-time intervention.
- Final untracked-code hash records published code and inherited dependencies, not a retroactive source snapshot of every run. Primary stage boundaries were retained while optional capture/profiler/envelope diagnostics were added afterward.

Original repository staged/unstaged diff binary hashes match the pre-task snapshots. Only this independent project is committed on the new branch; no model weights/images/hidden state/profiler database upload. Resource handoff overrides the earlier burner-restoration policy; allowed GPUs are empty at the recorded snapshot and no further GPU run is required.

Post-commit verification: a clean detached checkout of evidence commit `fc5ed196b83a75c34f0324c2b53affffe1c6b69e` ran the8 tests, the complete compact-bundle verifier and all five figure regenerations **without local raw runs/hidden captures**. All passed; checkout remained clean. This confirms fresh-checkout CPU usability, not a second physical GPU replication. Scoped `git diff --check` passes; the exact authoritative snapshot retains the source's intentional Markdown hard-break trailing spaces.

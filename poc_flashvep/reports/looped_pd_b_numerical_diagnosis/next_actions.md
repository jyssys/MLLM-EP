# Resume instructions

## Final original-task release

The numerical task is complete within its documented matched-runtime scope.
Ouro passes its tested200/400 repeat and B comparisons.
Huginn `down_norm_adapter` passes both200 and both400 repeat/B cohorts.
Final new-variant KV/hidden bytes, source-preserving order and batch1–16 geometry checks pass.
Read `state/final_release_checks.json` and `configs/final_runtime_hashes.json`.
Do not restart optional B campaigns or canceled controllers.
The new RCC-PD task must pin the final diagnosis commit and independently qualify its own mixed execution.
HF fidelity remains8/8 Ouro and5/8 Huginn on small triage only. Answer accuracy is unmeasured.

## Historical scope change: prepare for RCC-PD

The user does not require additional B experiments after baseline parity qualification.
The wait-only adapter postcheck and CPU postprocess controllers were stopped safely.
Their states record `NOT_RUN_USER_SCOPE_CHANGE`.
The existing finite GPU cohorts were not interrupted. Their qualification controller can finish and restore owned burners.
Do not restart the canceled follow-up controllers automatically.
Read the user's new RCC-PD specification before any RCC-PD implementation.
No RCC-PD implementation is authorized by the proposal discussion alone.

Ouro's tested200/400 internal gates pass.
Huginn adapter-corrected PD_G1 repeats pass both200 and both400 cohorts.
The independent200-request Huginn B comparison also passes.
Original-HF equivalence and benchmark answer accuracy remain separate, unproven gates.

The current task is numerical debugging and B transfer diagnosis.
Do not restart an old serving-capacity campaign.
Do not modify `ref_loopserve`, the installed RLT source, or frozen B reports.

## Current adapter correction, 2026-10-30

The active controller is `adapter_qualification.py`.
Read `state/huginn_adapter_qualification.json` and its recorded PID before any restart.
The new variant is `down_norm_adapter`.
It fixes the confirmed Huginn adapter geometry in addition to Down and RMSNorm.
Teacher replays of residual samples85 and56 pass batch1/2/4/8 with no inspected tensor difference.
Do not call that result a completed repair.

After the small N8 triplet passes, four physical pairs run finite numerical gates:

- GPU0,1: N200, rate0.2, old seed2026103001.
- GPU2,3: N200, rate0.2, failed independent seed2026109001.
- GPU4,5: N400, closed-loop capacity8, failed seed2026109001.
- GPU6,7: N400, closed-loop capacity8, new independent seed2026109002.

Each gate compares PD_G1, an independent PD_G1 repeat, and then B.
A baseline repeat failure stops that cohort before B.
These parallel runs provide correctness evidence only. Do not publish their latency as clean serving speedup.
After expanded gates pass, run a small new-variant KV fingerprint and preserved-seed order check.
Original-HF fidelity and benchmark answer accuracy remain separate, unproven gates.

## Completed earlier experiments

- Ouro original200 reproduction uses GPU0/1. It completed with181/200 repeat parity and179/200 B parity.
- Ouro `down_norm` passes the old and independent400-request seeds.
- Huginn original93 reproduction completed. Its repeat and B comparisons fail.
- Huginn `down_norm` passes the old200 seed but fails the new200 seed and the400-request repeat.
- The isolated transfer comparison, generic queued control and four quiet Nsight profiles completed.
- The previous `postprocess.py` pass completed before the adapter correction. Its manifest needs an update after new gates.

Read `state/` and per-run `status.json` before starting another process.
The controllers reuse COMPLETE/FAILED namespaces and refuse ambiguous RUNNING namespaces.
Never start two jobs on the same occupied pair.

## Findings to preserve

- Original trace failures are captured in `configs/failure_manifest.json`.
- Ouro actual-history replay first finds identical-input Down output differences during prompt execution.
- Down-only replay still finds identical-input RMSNorm differences during decode.
- Huginn's first valid traced difference is `core_block.2.norm_1`, input step22, depth14.
- An FP32 mean difference causes BF16 cast differences. The small-row reduction fix has a real-operand GPU regression.
- The old trace's repeated `ln_f` key is invalid. Use occurrence-indexed traces, or exclude that key.
- Do not read `topk_first` as greedy argmax when scores tie.
- The combined teacher replay is internally invariant on completed tested geometries. This is not the full PD acceptance gate.
- B's inherited `can_schedule` blocks subsequent prefill while queued transfer fragments remain.
- An opt-in queued-compute test retains producer events, pinned source memory, transfer limits and remote completion checks.

## Remaining work

1. Review the new adapter qualification gates. If another operation fails, preserve it and isolate that operation.
2. Verify KV/hidden byte fingerprints and full-depth work separately.
3. Run short isolated transfer tests and the queued-compute ablation. Do not report them as a capacity sweep.
4. Separate host API durations, exposed handoff and GPU DMA time.
5. Finish numerical, ablation, transfer and final reports in snake_case root files.
6. Compress large `result.json` files with `artifacts.py`. Preserve the originals locally.
7. Restore owned burners. Check unrelated staged-index preservation.
8. Commit only this task's source and report paths. Push normally and verify the remote SHA.

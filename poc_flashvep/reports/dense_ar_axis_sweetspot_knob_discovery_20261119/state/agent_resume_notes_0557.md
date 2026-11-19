# Research resume checkpoint

This checkpoint is not a final result report.

## Budget and preservation

- Start: 2026-11-19 01:14:22 KST. Do not reset this time.
- Deadline: 11:14:22 KST. Last new GPU start: 10:14:22 KST.
- Latest clock: 05:57:24 KST.
- Source: `poc_flashvep/dense_ar_axis_sweetspot/`.
- Evidence: this directory's parent.
- Base: `4cf3f9b80a2a8618da3bdddb1140fab7f403c3c7`.
- New branch: `flashvep/dense-ar-axis-sweetspot-knob-discovery-20261119`.
- User checkout remains the old dirty branch. Never use its index, checkout, reset or ordinary commit.
- Use `common.preserve()` and the private-index publisher.
- No subagents are authorized. Do not spawn them.
- New prose follows the ASD skill. Keep Korean. Do not announce its writing mode.

## Active controllers

Read current PID records before acting. Do not restart completed experiments.

- Main research: PID183063, `logs/research_campaign.log`.
- Batch extension: PID236047, `logs/batch_extension.log`.
- Output/QPS extensions: PID278161, `logs/counterfactual_extensions.log`.
- New batch-cost confirmation: PID466862, `logs/batch4_cost_confirmation.log`.
- Monitor v2: PID417566, `logs/monitor_v2.log`.
- Host-gap model helper: PID492995, `logs/freeze_control_gap_model.log`.
- Earlier initial, diagnostic, serial observer, Source Window and cost-v2 helpers completed.
- Monitor PID195585 was deliberately stopped and replaced after a CI-method correction.
- Shared quiet-lane lock serializes all GPU jobs. Do not start burners outside that lease during timing.
- Approved resource facade restores all eight burners after controllers finish. Explicit final closeout is still required.

The main controller completed 22/28 residency starts. It next runs 54 held-out starts.
The batch controller completed 17/18 performance starts after all six Batch4 state gates passed.
The extension controller then runs 18 Output-only tests and nine QPS3 tests.
The new batch-cost controller runs six calibration starts, freezes v3, then runs18 new confirmations.
Do not add further GPU grids. The planned total is241 successful dispatchers plus the preserved initial v1 failure.

## Measured mechanism evidence

All new actual-Batch state gates pass so far. B8192/S1024 is verified at both workers.
N36/N72 performance plans match aggregate GEMM rows exactly. N8/N32 are state tests only.

The actual Serving Linear API is `linear_batch_invariant`, not generic `F.linear`.
Weighted isolated Token/Layer D projection ratios are1.9103,1.1014,1.0671,1.0492 for Last2/4/8/12.
These are structural sums of isolated medians. They are not whole Forward or TTFT speedups.
64/72 isolated profiler gates pass. Eight small O-projection profiles remain Diagnostic.
All four online Event Observer cells failed the5% gate.
The bounded Serial alternative passed only for Layer, not Token. Do not expand profiling.
Exclusive online GPU components, SM/HBM counters and physical idle remain NOT_MEASURED.

The old Last4/8/12 tail differences are661.815,−1979.144,−3224.884ms.
A post-run max-stage-cycle×tail-rank approximation gives616.021,−2024.071,−3230.939ms.
This is an explanation using the same completed traces. It is not held-out prediction or causal proof.

The source Window32→4 intervention relocates much of Ready→D waiting to Arrival→P waiting.
Last12 has three paired seeds. Its CI is wide. Do not claim equivalence or a reliable speedup.

The first two new Last8 seeds show Layer wins at Decode2 and Token wins at Decode4.
These comparisons keep P1/D Suffix1 and the same partition. The third seed is still completing.
V2 extrapolated P costs incorrectly forecast the Batch4 rank. Preserve this failure.

## Important new root cause: reservation ACK feedback

Every P prefix waits for a D reservation ACK before submission.
D processes control messages only between `engine.step()` calls.
Credit rejection0 does not imply zero admission-control latency.

Fresh seed1930 Last8 backlogged, non-window-blocked gaps:

| Axis/Decode | P submit→ready | Next-P gap | ACK within gap | D guard/request |
|---|---:|---:|---:|---:|
| Layer/2 |225.69ms|26.77ms|24.85ms|254.29ms|
| Token/2 |235.55ms|29.61ms|27.73ms|275.86ms|
| Layer/4 |234.82ms|21.72ms|19.45ms|167.21ms|
| Token/4 |227.10ms|20.85ms|18.17ms|180.69ms|

Thus Decode2 is D-bound in the host-demand model. Decode4 is P-bound.
Add the gap to the P enclosure before deciding the effective bottleneck.
The original v2 extractor used a median gap and excluded gaps≥50ms.
This can understate bimodal reservation waits. Do not call these gaps GPU time.
`reservation_feedback.py` reconstructs exact gap components and logical D phases from existing logs.

## Frozen models and fresh validation

- v1 pressure rule: κ4, frozen from33 historical discovery cells.
- v2 action costs: frozen from30 new discovery starts. Never edit that file.
- v2 two-scalar rule: calibrated Decode demand d plus total unfinished n.
- Score: alpha×max(p,u+d)+(1−alpha)×(p+u). The equation is frozen.
- New v3: replace only Last8/O16/Decode4 costs using six new calibration starts, seeds1960/1961/1962.
- v3 evaluation: new seeds1970/1971/1972, QPS3 and6, Native/Layer/Token, actual1/1/4.
- v4: same six calibration starts. Replace only P_gap with the median of three per-start uncensored, backlogged, non-window-blocked mean gaps.
- v4 is a separate file. The helper must freeze it before any new confirmation measurement begins.
- v4 protocol was written at05:52 KST, before the six calibration starts.
- Do not refit v2/v3/v4 or the state rules after these confirmations.
- `evaluate_models`, `feature_validation` and `step_prediction_validation` now support versions2/3/4.
- Versions3/4 are evaluated only on the new batch4 confirmation stage.

## CI correction

The old mean CI interpolated27 exact bootstrap means for n3.
The new `resampling.py` uses inverse empirical-CDF quantiles and preserves endpoint mass1/27.
One Source Window example changed CI upper0.98547→1.00002 without changing raw values.
Old derived indexes were copied to `analysis/bootstrap_CDF_correction_prior_derived/`.
Frozen historical reports and raw measurements were not changed.
Monitor v2 reloads the corrected statistics module. All new analyses use the same corrected CI method.
Tests currently total39. Run the complete list again at closeout.

## Offline oracle scope

Old block oracle uses18-request blocks, or36 for Decode4.
`cohort_axis_oracle.py` adds a finer exact-work search.
One axis is fixed within each actual-size Decode cohort.
DP tracks Token-row debt and closes aggregate work exactly at the trace end.
Beam widths8/32 are finite searches, not global upper bounds.
The larger search includes the smaller winner and both Fixed actions.
Same-score causal shadow rules read no future arrival or completion.
They do use the predeclared finite homogeneous request-index plan for budget closure.
This bookkeeping is not production-stream generalization.
Oracle/rule GPU mixed-axis correctness is NOT_RUN. No online Dynamic Scheduler was implemented.
Fixed-trace timing and SLO validation must pass before interpreting headroom.
Mean Decode intervals cannot establish ITL-tail safety.

## Remaining work

1. Continue monitoring all controllers. Check critical-stop records and preserve all failures.
2. Verify the v3/v4 freeze before new confirmations. Verify all experiments meet exact state/work/batch gates.
3. Do not add GPU runs. Retain the10:14/11:14 limits.
4. When all GPU controllers and monitor/helpers exit, run `final_checks`.
5. Run `analysis_pipeline`. It refreshes all analyses and figures under the quiet lease.
6. Add/review final figures for v3/v4, ACK feedback, decision boundaries and fine-oracle headroom. Current figure code does not yet show every new v4 analysis.
7. Run `reports --final`. Review scientific conclusions and actual numbers. Do not accept template prose without inspection.
8. The report generator writes seven new reports. The independently written Phase A report stays intact.
9. All15 user questions must have explicit answers. Include v2 failures, v3/v4 fresh results, CI method and cost-model uncertainty.
10. Use `resource_closeout` to restore all eight approved burners and verify three positive-usage samples.
11. Run `publication --archive`, `--prepare`, `--push`. No real commit/push was done yet.
12. Verify remote SHA, user index/checkout, external runtime hashes and burner processes after push.
13. Leave burners running. Final response should be concise Korean with results, report path, branch/SHA and resource status.

Scripts are modules under `poc_flashvep.dense_ar_axis_sweetspot`.
Use `/home/esjung/.venvs/looped-pd-abc/bin/python`.
The final report is `poc_flashvep/reports/dense_ar_axis_sweetspot_knob_discovery_final_summary.md`.
Do not claim Sustainable Capacity, exclusive GPU service, physical idle or online Dynamic speedup from these finite controls.

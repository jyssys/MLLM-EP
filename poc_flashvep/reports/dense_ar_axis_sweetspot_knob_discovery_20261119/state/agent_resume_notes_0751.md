# Resume checkpoint — 2026-11-19 07:51 KST

The research budget still starts at01:14:22 KST. The deadline is11:14:22 KST.
Do not start a new GPU run after10:14:22 KST. Do not add more GPU grids.
The declared total is241 successful dispatchers plus one preserved initial failure.
At07:50:50,186 dispatchers completed. No new critical stop exists.

## Active processes

- Main research controller PID183063. Its held-out stage has54 runs.
- Counterfactual controller PID278161. Output intervention18/18 completed. Its new-QPS stage has9 runs.
- Batch4 calibration/confirmation controller PID466862. Calibration6/6 completed. Confirmation18 runs remain in progress.
- Monitor v2 PID417566 remains active.
- The v4 freeze helper and C1 freeze helper returned successfully.

All controllers share the existing quiet-lane lease. Do not launch duplicate jobs.
The monitor updates `state/resume_progress_manifest.json`.
Controller logs show exact per-stage progress. The shared research-progress file can name another controller's stage.

## Preservation and publication

Keep the user's checkout and index unchanged. The user HEAD isf4ecea59f36f620c69c4fef09c590bf790802cd9.
The new branch exists but is not checked out.
Use the private-index publisher in `poc_flashvep/dense_ar_axis_sweetspot/publication.py`.
No actual commit or push occurred yet.
The evidence namespace is `dense_ar_axis_sweetspot_knob_discovery_20261119`.

`snapshot_integrity.py` now checks742 inherited Python sources.
The initial source ZIP and current inherited sources equal base4cf3f9b80a2a8618da3bdddb1140fab7f403c3c7.
It excludes new campaign code and the generated top-level sitecustomize bootstrap.
The permanent receipt is `analysis/inherited_snapshot_integrity.json`.
Final checks repeat this audit.

## Newly completed controlled Output crossover

Last4, exact matched Token work, Input8000, QPS6, actual1/1/2, N72, three paired starts:

| Output | Layer/Token TTFT p95 |95% CI | Layer/Token qualified token/s |95% CI |
|---|---:|---|---:|---|
|16|1.0335405885|[1.0021345082,1.0801183061]|0.9620426152|[0.8969736310,0.9991951530]|
|64|0.9817489111|[0.9678727932,0.9926665821]|1.1068001975|[1.0255809870,1.1570894077]|

Both axes pass Native-relative5% for ITL p99 and TPOT p95 at both outputs.
These are finite overloaded cohorts. They are not sustainable capacity measurements.
TTFT improvements are small. The result is not a dynamic-scheduler speedup.
The Output cap also changes derived max_model_len8016→8064. Do not claim an isolated GPU-Decode-only intervention.

Mean client-TTFT identity, Layer−Token:

- O16: meanTTFT+166.77ms. Arrival→P+190.25ms. Pservice+3.96ms. Ready→D+2.97ms. D→firstWorker−38.44ms. Worker→Client+8.03ms.
- O64: meanTTFT−414.75ms. Arrival→P−75.38ms. Pservice+2.87ms. Ready→D−317.51ms. D→firstWorker−33.15ms. Worker→Client+8.42ms.

Do not add stage-p95 values. These are mean identities, not independent causal effects.
The Output intervention increases D Decode residence by about558ms/request.
P Service, ACK and D preparation changes have CIs that include0.
`causal_decomposition.py` now exports these interventions and new mean-identity summaries.
The report generator includes both tables.

## Knob evidence so far

Recorded feature validation now includes the Output crossover.
The pressure rule predicts Layer for both outputs. It misses the O16 Token winner.
The two-scalar rule predicts Token atO16 and Layer atO64.
Static Metadata Cycle makes the same correct predictions. It does not use changing within-workload state.
Therefore this success does not establish a dynamic contribution.
The report explains Static Metadata Cycle as the scalar:
`DeltaC=max(pL,uL+d)-max(pT,uT+d)` with threshold0.
This is calibrated host demand, not physical GPU slack.

The earlier Last8 Decode2 three-start result still favors Layer:
TTFT ratio0.88836935 CI[0.80938663,0.97790381].
Qualified token/s ratio1.15821460 CI[1.04228068,1.28714723].
Decode4 remains unresolved: TTFT ratio1.03078202 CI[0.95731354,1.11748894].
Keep the third-seed reversal. Do not call Token the confirmed Decode4 winner.

## Cost models and final bounded replay

Models v2, v3 and v4 are frozen. Do not refit them.
Both v4 and the explicit-control C1 inputs froze before any new1970/71/72 measured arrival.
Final checks verify their SHA256 and monotonic-clock ordering.
The C1 inputs are `configs/frozen_control_feedback_replay_inputs_v1.json`.

First new seed1970:

- v4 passes both fixed actions atQPS6. It fails Layer timing atQPS3.
- C1 passes Layer atQPS3/6. It fails Token timing at bothQPS values.
- Token C1 ACK signed mean error is about+18~19ms.
- Both C1 mixed searches remain NOT_RUN because fixed validation fails.
- Do not tune host-step boundaries after these failures.

`calibration_sensitivity.py` uses only the six new calibration traces.
Effective-P paired mean Layer−Token is+3.8066ms, CI[−4.2934,+10.1287]ms.
The separately fitted v4 medians differ by only about8microseconds.
The cost rank is not robustly identified. Do not present this small forecast difference as reliable.
The permanent artifact is `analysis/new_calibration_cost_rank_uncertainty.json`.

## Verified code and figures

All44 CPU tests passed at06:41. Recheck at final closeout.
Reports now tolerate both early and later decision-log schemas. Preserve the original journal.
Seven report drafts were generated successfully. They remain ACTIVE_DRAFT_NOT_FINAL.
The independently written PhaseA report is not overwritten.

Figures were generated successfully. Representative figures were visually inspected.
New explicit-control replay error figure exists. It distinguishes offline error from measured speedup.
The final pipeline regenerates all figures and reports after the GPU phase ends.
Review final prose and all numeric claims manually. Update the summary with the new controlled crossover.

## Final sequence

1. Wait for the three active controllers and monitor to return. Inspect all stop/failure markers.
2. Run `final_checks`. It verifies preservation,44 CPU tests, completed gates, frozen models and snapshot integrity.
3. Run `analysis_pipeline`. It includes calibration sensitivity and C1 replay validation.
4. Generate final reports with `reports --final`. Review figures and scientific claims.
5. Run `resource_closeout`. Verify approved burners on all8 GPUs with three positive NVML samples.
6. Run private-index publication selftest, archive, prepare and push. Verify remote SHA.
7. Verify preservation and all8 burners again. Leave burners running.

All reporting uses short sentences and the user's requested language. Do not announce the writing mode.
User updates remain minimal. Do not claim a universal1–2-knob rule or a validated5% dynamic gain without the final evidence.

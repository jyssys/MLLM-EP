# Reviewed campaign checkpoint

The discovery ended with BLOCKED advancement at Phase 3/4.
No experiment worker remains active.
Owned burners run on all eight devices.
`research_state.json` records the final status.

The canonical result is `analysis.md` with the reviewed table in `phase_gates.md`.
Do not dispatch A1/A2/A3 from the current predictor.
The aggregate bound hides 79.141% Atomic-prefill and 90.901% staged-prefill coverage.
The stock-policy completed-token target also lacks candidate-specific rollout validity.

The next smallest experiment requires a new registration and fresh source families.
It should test path-stratified bounds and pre-selection event readiness.
It should then check a bounded complete-token schedule against independent live timelines.
Do not refit on the published HELDOUT errors or delete blind results to bypass the fit guard.
The paused native-vLLM PP program remains excluded.

## Archived earlier checkpoint

The following queue instructions describe the earlier running state.
The listed queues are now complete.
Do not interpret them as pending work.

## Running queue

`quality_and_surface_dispatch_v1.json` contains eight quality jobs and 96 main cost jobs.
The dispatcher owns a file lock in `state/dispatcher.lock`.
Do not start another dispatcher while it runs.
Inspect `run_manifest.jsonl`, `state/gpuN.json`, and completed-result hashes.
The original approved GPU set is now physical 0–7.
These are independent single-GPU workers.

The queue begins with the 114-case quality pilot per policy.
Each GPU runs Atomic, Grouped, and Mixed on its frozen question block.
The main cost jobs can move to an idle worker after their preferred device is busy.
All six policy cells remain paired within one physical GPU/process-start block.

## Qualified source and completed checks

The active RLT checkout is `/home/esjung/external/ouro-deadline-g3-candidate-rlt`.
Its pin is `4a350344027960a48384626aa646f1765fda6d4f`.
Its diff hash is `1feb47c40a10e2f6c0cc9a314aec86ff96c6b251954eaed31d1be0b859f67862`.
The original calibration checkout remains intact.
The candidate fixes a reproduced G3/G1 execution-boundary batch error.

All 64 contention jobs completed.
The drift is not uniformly neutral.
Read `gpu_contention.md` before clean serving claims.
All 160 isolated numerical comparisons are exact.
All 48 preflight cohorts pass full-R4 accounting.
Their maximum recorded event-clock alignment bound is 0.058981746435165405 ms.

The targeted `bg_11_3` mismatch is attributed to BF16 GEMM batch geometry.
The same MLP input and weights reproduce Atomic M1024 and Grouped M48 outputs exactly.
Read `first_divergence.md` and the operator replay raw result.
Other mismatches and full-answer quality remain separately qualified.

## Analysis sequence

1. Finish the current queue without repeating completed jobs.
2. Run `grade_quality` after all eight quality jobs finish.
3. Run `verify_surface --pattern 'p2_P*'` for the completed main grid.
4. Fit quantum and full-next-token predictors from disjoint source/run partitions.
5. Select models on VALIDATION only. Fit safety margins on CALIBRATION only.
6. Report HELDOUT error and coverage without threshold relaxation.
7. Analyze legal C-only/G-only/Joint action headroom before any online implementation.

The command prefix is `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.`.
The analysis modules are `grade_quality`, `verify_surface`, `predictor`, and `decode_prediction`.
The predictor extractor excludes harness/preflight files.
Current observer schemas 1/2/3 remain distinct.
Missing fields remain explicit. Do not fill absent measurements with invented values.
Do not change the worker or observer during the remaining queue without a documented new lane.

## Remaining research gates

The legal-action and bounded next-token DAG modules are definition tools only.
No Phase 4 measured headroom report exists yet.
Native logical loop progress can precede GPU completion.
Pending GPU events must enter a deadline model.
Do not sum overlapping intervals or use future measured durations online.

Before Phase 5, compare against tuned C-only and unmodified Mixed.
Preserve the staged-G4 control.
A prettier G2 P99 alone does not justify adaptive implementation.
Keep the final online source partition untouched until the predictor and policy freeze.

## Safety and publication

Use the scoped `resources` helper for owned burners.
It explicitly passes the GPU mask because `run_utilize.sh` has an outdated default.
Empty finished devices receive burners automatically.
Do not stop unrelated jobs.
The native-vLLM Ouro PP program remains paused.

The original 68 staged additions remain protected.
Their binary diff hash is `7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9`.
Commit only the new package and report directory.
Large tensors and profiler captures stay local with hashes.
No final campaign push is complete yet.

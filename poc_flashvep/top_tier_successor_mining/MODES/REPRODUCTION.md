# MoDES reproduction status at GPU release

## Final GPU resumption supersedes the historical checkpoint below

Final resumption: official1024-question/grid100 frontier completed,436 unique evaluations. Held-out256-question quality and corrected six-condition EP screen completed;three independent GQA-B16 engines do not reproduce the initial large positive point. Known OCR trade-off and port-cost floor do not establish a nontrivial successor. Kimi and new prototype not run.

## Historical audit / release record

ALGORITHM_SANITY_PASS; IMPORTANCE_CALIBRATION_DONE; FRONTIER_PARTIAL.

Original mask/scaling and normalized-input ablation behavior were tested against
official source. Public zero-weight simulation is not treated as fast inference.
Existing DeepEP invalid-ID sentinel primitive tests pass; updated BF16 decision
and clean serving port still require the deferred fresh verification.

Full1024 GQA calibration completed with original raw-question/full-answer mask
and48-layer×2-modality ablations. The100-grid original search is interrupted:
301 unique measured threshold pairs,70% target completed;85% target unfinished.

Completed70% target: actual skip71.3284%, KL.0110761. Best measured85%-feasible
point so far: actual85.2668%, KL.0170227. The latter is NOT certified as the final
official search optimum. Both statistics are calibration, not held-out accuracy.

Pilot held-out results used smaller/coarser calibration. Do not retroactively
label their predictions as outputs of the full official calibration.
# 2026-09-08 official calibration completion

The interrupted1024-example, grid100 search completed on physical4–7 after
exact cached-point parity. It evaluated436 unique threshold pairs in total.
Resumed four-replica process interval:3140.673s including reload/preparation,
not EP timing and not an assertion of CUDA-active duration.

| Target skip | Actual calibration skip | Answer-token KL | tau text | tau vision |
|---|---:|---:|---:|---:|
|70%|71.32836%|0.01107609|0.000129953750026|0.00388271624604|
|85%|85.26683%|0.01702272|0.0000907957374049|0.00525341508107|

Both optima match the earlier partial bests, now certified only with respect to
the original frontier algorithm and this discrete search, not a global optimum
over arbitrary thresholds. Held-out quality and clean EP performance remain
separate subsequent gates. Historical partial-search notes below are retained.

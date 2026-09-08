# SERE reproduction status at GPU release

## Final GPU resumption supersedes the historical checkpoint below

Final resumption: official calibration,256-question HF confirmation and six natural EP conditions completed. Existing S4 repairs100%/88.9% of observed B1 quality loss. Fixed-length/no-op controls isolate port cost. INCREMENTAL_ONLY; original-paper speed and Kimi not reproduced. See BEST_CANDIDATE_DEEP_DIVE.md and the final main report.

## Historical audit / release record

ALGORITHM_SANITY_PASS; MLLM_TRANSFER_EXPLORATORY; CLEAN_SERVING_DEFERRED.

The original CUDA rerouter passed128 route identity tests; weights/duplicate IDs
remain unchanged. HF zero-threshold/no-op controls pass. The real EP4 sanity
observed the official rerouter plus DeepEPHTPrepareAndFinalize/TritonExperts.
It was a tiny instrumented run, not an efficiency reproduction.

400×128 FineWeb calibration was performed twice: exploratory FP32-Gram and
confirmatory rounded BF16 subtraction/norm. The latter table is ready, but its
task-quality/serving evaluation was deferred by GPU release. Current quality
numbers use the exploratory table and must retain that label.

Original paper: single-H20, vLLM0.8.4/V0, Qwen3-Instruct2507 among models, stochastic
batch16 quality and fixed128/32 speed setting. Current transfer: Qwen3-VL BF16,
greedy short answers, HF quality replicas and bounded vLLM0.20/V1 EP4 port.
No claim of matching original numerical benchmark scores.

See BASELINE_TRANSFER_CONTRACT.md for batch-union semantics and natural-EOS vs
fixed-output separation. Kimi and clean request-speed measurements did not run.

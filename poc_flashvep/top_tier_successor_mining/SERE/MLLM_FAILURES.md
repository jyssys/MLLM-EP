# SERE exploratory failure matrix

## Final GPU resumption supersedes the historical checkpoint below

Final resumption: official calibration,256-question HF confirmation and six natural EP conditions completed. Existing S4 repairs100%/88.9% of observed B1 quality loss. Fixed-length/no-op controls isolate port cost. INCREMENTAL_ONLY; original-paper speed and Kimi not reproduced. See BEST_CANDIDATE_DEEP_DIVE.md and the final main report.

## Historical audit / release record

128 paired ChartQA images: vanilla89.0625%; S2/rho.5 all or decode-only69.53125%
(-19.53125pp). Prefill-only88.28125% (-.78125pp, CI crosses0). This localizes the
pilot loss to generated-text decode, not to visual prefill alone.

Batch16 vanilla89.84375%, S2/rho.5 88.28125% (-1.5625pp,95%CI[-3.90625,0]). S4
or rho.7 further recover quality. Thus batch/threshold is a strong trivial-fix
alternative. Retained speed at those safer settings is unmeasured.

Natural mean output length4.63→6.62 tokens for all-phase rerouting,7.24 for
decode-only. First-line-only scoring cannot recover all errors; retain official
full-prediction scoring. This is not yet evidence of an E2E latency penalty.

New CPU-only sensitivity: 6,336 route conditions across24 captured requests and
eight layers. At S2/rho.5, changing calibration arithmetic changes roughly
.85–2.31% of assignments across sampled populations. These are sampled PREFILL
routes, NOT decode captures. Even a small route change may affect greedy output;
the pilot is not promoted without confirmatory generation.

Classification: unconfirmed material small-batch transfer loss with strong simple
controls, not a proven MLLM-specific/nontrivial successor failure.

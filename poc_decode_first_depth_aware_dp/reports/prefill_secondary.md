# Prefill secondary audit

Verdict: **PREFILL_FULL_DEPTH** for the tested official CDB path.

The engine executes full-depth prefill and samples the first output token from that result.
Adaptive staged decode begins only after the first generated token.
The first token therefore does not enter adaptive-decode depth statistics.

The small probe has 13 prompt tokens.
The runtime counter records 52 prefill recurrent token-steps in both policies.
The layer observer records four complete 24-layer prefill passes.
The subsequent decode work differs, but the prefill work does not.

The quality runner also asserts `prefill_recurrent_steps == 4 * prompt_tokens` for each request.
These assertions preserve the distinction between full-depth prefill and adaptive decode.
The runner does not relabel a prefill-produced first token as an early-exited decode token.

The source path is `ContinuousDepthBatchingEngine` with `ModelRunner.stage_prefill_batch`.
The staged recurrent adapter serves the later decode tokens.
The engine documentation and implementation both specify full-depth prefill.

The study establishes no prefill-specific depth placement advantage.
Prompt length remains a possible generic admission feature.
The study does not construct a token-selective prefill runtime.
It does not infer that all looped models or MoR designs are decode-only.

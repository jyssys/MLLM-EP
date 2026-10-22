# Correctness and quality scope

## Isolated numerical comparisons

The new probes compare Atomic with Split, Grouped, and Mixed.
They cover C128/256/512/1024 and prompt lengths 128/512/2048/8192.
Each mode generates eight actual greedy tokens at full R4.
The probe reads every 24-layer × four-depth KV namespace before generation completes.
It compares the final prompt hidden state and LM-head logits.

All 48 non-reference comparisons passed exactly.
Output tokens, exit-depth records, hidden states, logits, and K/V contents matched.
Maximum recorded KV relative L2 error was 0.
The KV comparison occurs at full-prompt completion, not after every decode token.
The table is `numerical_probe_summary.csv`.
These probes are not full-answer benchmark accuracy.

The original numerical limits remain unchanged.
The isolated exact matches are stronger than a tolerance-only pass.
They do not resolve the separate batched mismatch evidence below.

The closed-cohort audit counts issued prompt work from actual submitted spans.
It checks four recurrent passes per prompt token.
The response surface records this assertion per measured cohort.
Static R4 applies to decode as well as prefill.
G1/G2/G4 changes scheduling granularity, not execution depth.

## Batched sequence differences remain unresolved

The historical Grouped-versus-Split batched match remains 112/120.
All eight mismatches affect background decode requests.
Their first divergent token indices are 99, 115, 91, 81, 96, 127, 28, 8.
The earlier C/G study did not establish full GSM8K/MATH accuracy.
This study does not retroactively convert those results into accuracy PASS.

`historical_mismatch_batch_audit.csv` aligns token and submission histories.
Some coda batch sizes and request memberships differ.
At the divergent interval, only one of eight cases immediately changes recurrent row counts.
Thus batch-size descriptions alone do not prove every mismatch cause.

## Targeted hidden/logit diagnostic

The diagnostic replays the ninth output of `bg_11_3` under Split and Grouped.
Both executions share the preceding eight output tokens.
The captured hidden-state relative L2 difference is 0.015339084900915623.
The native emitted token is 216 for Split and 909 for Grouped.
The captured coda has one actual row in both executions.

The diagnostic applies the same LM head to each captured hidden vector.
BF16 row widths 1/2/4/7/8 preserve that vector's argmax-to-FP32 agreement status.
Split has a BF16 top-two margin of 0.
Grouped has a BF16 margin of 0.125.
Both FP32 diagnostic argmax values are 909.
This shows sensitivity near a quantized logit tie.
It does not isolate the earlier source of the hidden-state difference.

The original raw `top1` field means the first index returned by `topk`.
It is not necessarily greedy `argmax` when logits tie.
The raw record also preserves native `original_argmax` and `same_argmax_as_FP32`.
Those fields must not be conflated.
The instrumentation now labels that distinction explicitly for future captures.
The frozen raw diagnostic remains unchanged.

This is an LM-head diagnostic replay, not a candidate generation method.
No FP32 operation replaces BF16 model arithmetic in serving.
It does not prove an output-quality advantage for either policy.
It does not exclude an earlier graph, lifecycle, or stream-related cause.

## Natural serving and quality

Natural serving uses the official tokenizer chat template and greedy generation.
The native EOS is token 0.
The frontend also stops after the actual generation of chat-turn end token 2.
The primary complete-response cap is 512 tokens.
A length-capped response does not qualify as a complete natural response.
All offered requests remain in the SLO denominator.

Natural output length and token content can differ between policies.
Therefore natural-serving latency is not an equal-output-work replay speedup.
The bounded128 continuity lane and the complete-stop lane remain separate.
`bounded_output_parity.csv` gives new matched bounded-output counts.
`request_events.csv.gz` retains natural tokens, times, depth, and stop reasons.

Full GSM8K/MATH benchmark accuracy is `NOT_RUN` in this characterization study.
The public conversational/article cohort is not a reasoning benchmark substitute.
Complete EOS or chat-stop is not proof of factual or mathematical correctness.
No benchmark noninferiority or universal token-exact parity is claimed.
Unresolved batched-output differences prevent a quality-qualified adaptive GO decision.

## Lifecycle and metric tests

The pinned RLT `tests/test_split_prefill.py` suite completed with 33 passes and 28 skips.
It ran with `CUDA_VISIBLE_DEVICES=''`.
Its tiny CPU fixtures check software state behavior, not pretrained GPU fidelity or benchmark quality.
`cpu_lifecycle_tests.json` preserves the command, scope, and complete test output.
Hardware cancellation, request-ID reuse under pressure, and all graph-buffer race paths are not exhaustively validated here.

Seven local metric tests check all-offered denominators, separate SLO outcomes, caps, timeouts, and empty responses.
Their synthetic timestamps are definition tests, not measured serving results.
The evidence-completeness audit separately checks expected jobs, device identity, and full-R4 work invariants.
It does not convert unresolved output differences into correctness PASS.

# Stage 1: online decode depth

The new pilot contains 128 distinct requests. Each of four domains contributes 32 requests.
Each domain has eight calibration requests and 24 held-out requests.
The metadata split preceded new inference.
The pilot uses GSM8K, MATH-500, Alpaca, and ShareGPT.
The discovery math questions exclude the old quality-pilot IDs.
Exact prompts, sources, revisions, and response hashes appear in `discovery_frozen.json`.

The measurement contains 31,946 real online decode exits.
The original gate selected R2 for 3,789 tokens, R3 for 24,438, and R4 for 3,719.
The first output token uses full-R4 prefill and is absent from this histogram.
The request-mean depth range is 2.4745 to 3.8667.

| Domain | N | Mean depth | 95% request-bootstrap CI | Between-request SD | EOS / N |
| --- | --- | --- | --- | --- | --- |
| alpaca | 32 | 2.9811 | [2.888, 3.077] | 0.2703 | 2/32 |
| gsm8k | 32 | 3.0979 | [3.072, 3.126] | 0.0781 | 0/32 |
| math500 | 32 | 3.0263 | [2.997, 3.056] | 0.0849 | 5/32 |
| sharegpt | 32 | 2.9179 | [2.841, 2.991] | 0.2178 | 0/32 |

The cap is 256 with natural EOS. We did not force continuation after EOS.
Only seven requests reached EOS within this bounded discovery run.
This pilot is not full-response benchmark quality evidence.
Equal caps do not imply equal actual output lengths.
The equal-progress analysis uses only requests that naturally reach each selected window.

## Cost relevance

| Retrospective features | Held-out N | MAE, ms | R² |
| --- | --- | --- | --- |
| length_prompt_domain | 96 | 148.214 | 0.8712 |
| plus_actual_depth_work | 96 | 20.814 | 0.9926 |

These models predict trace-lane B1 decode duration.
Actual loop work is a retrospective feature, not an admission-time feature.
The comparison controls prompt length, output length, and declared domain.
It does not establish that a prompt can predict future depth.
It also does not establish a DP scheduling gain.

The physical heterogeneity gate passes for this pilot.
The campaign does not expand to 512 requests before resolving the predictor and baseline limitations.

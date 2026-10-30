# Correctness and original-reference fidelity

The new sequential qualification uses 40 held-out ShareGPT prompts per model, greedy sampling, natural EOS, and output cap32.
The controls share input token IDs, seed, fixed recurrence, precision, cache layout, and token chunk256.
These diagnostic jobs run concurrently across separate GPU pairs. Their latencies are not clean serving results.

| Model | PD_G1/B exact sequences | Equal depth traces | Valid transferred KV/hidden identity |
|---|---:|---:|---|
| Ouro R4 | 40/40 | 40/40 | PASS on the separate instrumented two-request run plus warmup |
| Huginn R32 | 40/40 | 40/40 | PASS on the separate instrumented two-request run plus warmup |

The byte checks retain reversed-fragment order and producing-event dependencies.
Huginn's prelude, coda, final-token handling, and RNG state use the existing qualified adapter.
No gate, weight, precision, attention equation, or recurrent depth changes.

Fresh eight-prompt official-HF triage gives Ouro7/8 and Huginn6/8 exact sequences.
Ouro's first mismatching output position is2. Huginn's mismatches begin at positions2 and5.
The denominator is only eight selected diagnostic prompts per model. It cannot estimate benchmark accuracy.
It does reproduce a remaining original-reference fidelity gap.
The prior frozen 100-prompt evidence remains Ouro88/100 and Huginn89/100.
Those prior counts are not merged with this diagnostic sample.

The HF reference uses Transformers4.57.6 and checkpoint-owned code.
The main RLT environment uses Transformers5.16.1 and FlashAttention4.
The new triage does not measure per-step logit error or prove that a discrepancy is harmless BF16 rounding.
The causal source of the mismatch remains unresolved.
The global status is `FIDELITY_UNRESOLVED`. No original-model-lossless GO is permitted.

The controller also compares B/PD_G1 output sequences on each held-out matched serving trace.
A method-dependent mismatch disables later B performance acceptance for that model.
Its already measured timing remains diagnostic evidence, not equivalent-work speedup.
GSM8K and HumanEval answer accuracy are NOT_MEASURED in this follow-up.

Evidence: `configs/qualification.json`, `configs/final_correctness_gate.json`, `configs/reference_comparison.json`, and per-run request/KV files.

## Subsequent W1 serving gate failure

Ouro failed the matched natural-serving gate after the initial qualification.
The first W1 point uses 200 requests, nominal arrival rate1request/s, seed2026103001, and output cap128.
Both arms use physical GPUs0,1 with the same P1:D1 split.

| Observation | PD_G1 versus B | Evidence type |
|---|---:|---|
| Identical input sequences | 200/200 | MEASURED sequence comparison |
| Identical generated sequences | 185/200 | MEASURED sequence comparison |
| Identical output lengths | 198/200 | MEASURED sequence comparison |
| Identical recorded depth traces | 198/200 | MEASURED sequence comparison |

The first five divergent samples are3,31,45,53,62.
Their first divergent output positions are125,54,41,42,8, respectively. Positions start at0.
The two unequal depth-trace lengths do not show early exit.
Every recorded depth value remains4. Different generated lengths cause the trace-length difference.

The controller closed Ouro's B serving-performance gate.
Subsequent B diagnostics cannot reopen this gate without a new qualification and an explicit experiment epoch.
The recorded W1 timing is output-confounded. It cannot establish lossless speedup or accepted capacity.
The current evidence does not distinguish batch-dependent BF16 divergence from an unrecognized state error.
Separate repeated-control and sequential mismatch-subset diagnostics are planned after clean serving.

Evidence: `state/ouro_B_serving_parity_failure.json` and the two matched W1 raw directories.

Huginn subsequently fails its first matched W1 serving gate too.
The point uses200 requests, nominal0.2requests/s, seed2026103001, output cap128, and GPUs0,1.

| Observation | Huginn PD_G1 versus B | Evidence type |
|---|---:|---|
| Identical input sequences | 200/200 | MEASURED sequence comparison |
| Identical generated sequences | 193/200 | MEASURED sequence comparison |
| Identical output lengths | 198/200 | MEASURED sequence comparison |
| Identical recorded depth traces | 198/200 | MEASURED sequence comparison |

Every recorded Huginn depth value remains32.
PD_G1 generates23621 tokens. B generates23603 tokens.
The controller closes the Huginn B serving-performance gate.
The first five divergent samples are30,53,120,121,157.
Their first divergent output positions are53,36,49,12,20, respectively.
These differences do not establish their cause or downstream answer accuracy.
Both models now require a new correctness qualification before any B speedup can be accepted.

The official-HF diagnostic records do not contain token-level `exit_depths` arrays.
A zero `depth_equal` count in that diagnostic comparison means missing trace evidence.
It does not show that HF used a different configured recurrent depth.
The helper explicitly sets Huginn `num_steps=32`. Ouro's pinned config sets `total_ut_steps=4` and `early_exit_threshold=1.0`.

## EOS accounting

The pinned RLT scheduler records an EOS stop as `FinishReason.STOP`, not the string `eos`.
It records the safety output cap as `FinishReason.LENGTH`.
The analysis preserves the raw reason and counts `stop` as EOS for this configuration.
This corrects a report-accounting label. It does not change generation or the frozen serving runtime.
Source: `vllm_rlt/core/scheduler.py`, function `_record_token`, lines1194–1202 in the pinned private checkout.

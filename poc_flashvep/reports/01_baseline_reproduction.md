# Baseline reproduction

Status: `PILOT_COMPLETE_NOT_FULL_REPRODUCTION`. The user deferred full fresh evaluation.
Model revision: `dad945cac317da394b390f82c7b40691d8a881ed`.
Evidence: `../results/refinement_aware_ep_expert_allocation/`.

The current scope is 128 GSM8K questions and 32 HumanEval questions per method.
See `refinement_ep_pilot_scope_amendment.md` for the user-approved change.
The original full-evaluation rows below remain unmeasured.

## Current pilot

| Dataset | Method | Correct/N | Accuracy | Wilson 95% CI | Difference from official |
|---|---|---:|---:|---|---:|
| GSM8K | Fresh HF, reused raw outputs | 118/128 | 92.18750% | [86.21531%, 95.70126%] | -2.05250 pp |
| GSM8K | Fresh physical EP4, reused raw outputs | 119/128 | 92.96875% | [87.17652%, 96.25702%] | -1.27125 pp |
| HumanEval | Fresh HF | 30/32 | 93.75000% | [79.85288%, 98.26894%] | +7.16000 pp |
| HumanEval | Fresh physical EP4 | 32/32 | 100.00000% | [89.28208%, 100.00000%] | +13.41000 pp |
| HumanEval | Fresh physical EP4, identity hook | 32/32 | 100.00000% | [89.28208%, 100.00000%] | +13.41000 pp |

The GSM8K EP4-minus-HF paired difference is +0.78125 pp.
Its question-bootstrap 95% interval is [-2.34375, +4.68750] pp.
HF and native token-selection protocols differ.
This difference is not a sharding-only accuracy estimate.
The small pilot does not prove 0.5 pp non-inferiority.
The HumanEval EP4-minus-HF difference is +6.25 pp.
Its paired bootstrap 95% interval is [0.00, +15.625] pp.
The native identity hook preserves token IDs and NFE on all 32 HumanEval requests.
This does not establish a new allocation method's quality.
The fixed first-32 subset and different decoder protocols prevent an official-score superiority claim.

## Published and historical accuracy

| Dataset | Source | Correct/N | Accuracy | Wilson 95% CI | Official reference | Difference |
|---|---|---:|---:|---|---:|---:|
| GSM8K | Published model card | NOT_AVAILABLE | 94.24% | NOT_AVAILABLE | 94.24% | 0.00 pp |
| GSM8K | Verified historical HF artifacts | 1211/1319 | 91.81198% | [90.20806%, 93.17306%] | 94.24% | -2.42802 pp |
| HumanEval | Published model card | NOT_AVAILABLE | 86.59% | NOT_AVAILABLE | 86.59% | 0.00 pp |
| HumanEval | Verified historical sandbox results | 143/164 | 87.19512% | [81.21751%, 91.47013%] | 86.59% | +0.60512 pp |
| GSM8K | Fresh HF full evaluation | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 94.24% | NOT_MEASURED |
| GSM8K | Fresh physical EP4 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 94.24% | NOT_MEASURED |
| HumanEval | Fresh HF full evaluation | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 86.59% | NOT_MEASURED |
| HumanEval | Fresh physical EP4 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 86.59% | NOT_MEASURED |

The historical rows are not new measurements.
The audit recomputed their counts from original per-request files.
`environment.json` records those file hashes.
The historical GSM8K gap remains unresolved.
No setting changes attempt to match the published score.

The [official model card](https://huggingface.co/inclusionAI/LLaDA2.0-mini) reports these reference scores.
Its example uses block length 32, steps 32, temperature zero, and EOS stopping.
The example alone does not identify the benchmark protocol.

## Frozen local protocol

GSM8K uses four prompt examples from the official dInfer Mini task YAML.
The generation cap is 2,048 new tokens.
HumanEval uses the original task prompt inside the official chat template.
Its generation cap is 1,024 new tokens.
Both paths use BF16, block length 32, threshold 0.95, and EOS stopping.

The HF implementation always calls `torch.multinomial` after softmax.
Temperature zero skips temperature scaling but does not select argmax.
The native threshold decoder uses greedy token selection.
The HF evaluation preserves its official source behavior.
It does not patch this discrepancy.
HF and native results therefore require separate protocol labels.

The HF path recomputes the visible prefix for each denoising forward.
The native path uses the existing prefix KV cache and cross-block refresh.
Its legacy NFE counter counts a complete prefill operation only once.
That operation can contain multiple model forward calls.
Postprocessing corrects this undercount from the source-confirmed prefill geometry.
The corrected field carries the label `SOURCE_DERIVED_PREFILL_COUNTER_CORRECTION`.
New worker starts count each model forward directly.
An equal NFE does not imply equal executed token work across these backends.
Within-policy comparisons must keep the native cache behavior unchanged.

The HF source also returns different tensor scopes on its early-EOS and ordinary paths.
The evaluator strips a prompt prefix only after an exact token-prefix match.
It records this normalization for each request.
It does not change model generation.

The native B1 return removes every EOS token from its output tensor.
The initial logger incorrectly called those shortened returns length-capped.
The original raw records remain unchanged.
Postprocessing derives EOS removal from the measured return length and audited source.
It labels this evidence `SOURCE_DERIVED_FROM_MEASURED_RETURN_LENGTH`.
New worker starts also capture the terminal token buffer directly.
This logging correction does not change generation or answer scoring.

## Scoring and limitations

The primary GSM8K parser matches the historical parser.
It prefers explicit answer markers and otherwise extracts the last number.
Every capped or invalid response stays in the denominator.
Termination-aware diagnostics must accompany the raw parser score.
A correct number in an unfinished response does not prove a complete answer.

HumanEval requires the existing networkless namespace sandbox.
A fresh sandbox smoke test passed.
Full completion execution remains pending.
Token agreement alone does not replace executable scoring.

The two fresh HF sanity requests completed.
The Group A EP4 sanity requests also completed.
These requests do not establish benchmark accuracy.
The full 1,319/164-question queues stopped at the user's request.
Their completed raw outputs remain preserved.
The reduced HumanEval pilot completed and passed sandbox scoring.
All methods retain capped outputs in the denominator.
The HF and native HumanEval pilots each contain one length-capped response.

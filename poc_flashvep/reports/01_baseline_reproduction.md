# Baseline reproduction

Status: `IN_PROGRESS`. Full fresh accuracy is not yet available.
Model revision: `dad945cac317da394b390f82c7b40691d8a881ed`.
Evidence: `../results/refinement_aware_ep_expert_allocation/`.

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
The full 1,319/164-question evaluations run in the durable queue.

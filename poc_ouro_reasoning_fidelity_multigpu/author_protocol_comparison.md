# Author protocol and public evaluation

This campaign uses the released Ouro-2.6B-Thinking checkpoint.
It does not reproduce the authors' private evaluator.
Protocol status is `REFERENCE_PROTOCOL_NOT_IDENTICAL`.

| Property | Published protocol | This campaign |
|---|---|---|
| Model | Ouro-2.6B-Thinking, R4 | Pinned revision `f1edd81e7ac41355db670500ceaf204e0f73af68`, R4 |
| AIME2024 | 30 problems, pass@1 64.7%, pass@10 90.0% | Same public 30-problem dataset, completed-answer exact match |
| Sampling | Temperature 1.0, top-p 0.7 | Same two values, explicit top-k=-1 |
| Prompt bytes | UNKNOWN | Frozen original instruction and pinned chat template |
| Thinking prefix | Not specified in the paper | P1 explicitly adds `<think>\n` |
| Sample count and seeds | UNKNOWN | Three seeds, 90 AIME outputs |
| Output cap for AIME | UNKNOWN | Complete-lane 8,192→16,384→24,576 rule |
| Stop rules | UNKNOWN | EOS enabled, no extra stop string |
| Answer grader | Private LLM judge and rubric | Public deterministic integer parser |
| pass@10 | Authors report this metric | NOT_RUN without ten samples per problem |

The paper's 8,192-token safety example does not define its AIME output cap.
The public exact-match score can differ from a private judge score.
Matching 64.7% alone cannot validate an implementation.
All missing author settings remain UNKNOWN.

The model card names Transformers 4.54.1.
The pinned repository also contains later cache and tokenizer changes.
The constructor probe records whether those components work together.
The campaign preserves the failed 4.54.1 run.
It does not downgrade or modify a shared environment.

Sources:

- [Paper, Section 5.2, Table 9, and Appendix C](https://arxiv.org/html/2510.25741v2)
- [Author project](https://ouro-llm.github.io/)
- [Pinned model card](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/f1edd81e7ac41355db670500ceaf204e0f73af68/README.md)
- [Merged tokenizer change](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/discussions/7)

The source manifest records retrieval status and SHA256 hashes.
The paper does not provide the exact private prompt, judge rubric, or seed stream.
This campaign cannot infer those settings from a desired score.

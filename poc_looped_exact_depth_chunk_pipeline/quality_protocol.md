# Native R4 quality protocol

The pinned checkpoint is ByteDance/Ouro-2.6B-Thinking at `f1edd81e7ac41355db670500ceaf204e0f73af68`.
All quality requests use four recurrent passes and `early_exit_threshold=1.0`.
The engine uses the official tokenizer chat template with `add_generation_prompt=True`.
The quality lane never forces output tokens. EOS remains enabled.

The primary benchmark contains all 30 AIME2024 problems.
Sampling uses temperature 1.0, top-p 0.7, and seeds 41015, 41016, and 41017.
The separate greedy lane uses temperature 0 and the same 30 problems.
MATH-500 contributes 100 questions through metadata-stratified hash selection.
Selection precedes all outputs. The secondary lane uses greedy generation.

The first output cap is 4,096 tokens.
If at least 10% of a complete lane reaches that cap, the complete lane repeats at 8,192 tokens.
Both cap results remain available. Capped outputs cannot count as completed correct answers.
At the final cap, each primary lane needs at least 90% natural termination for the completion gate.

Quality uses batch sizes up to eight for cost control. It is not the B1 hardware timing lane.
Each sampled request has an explicit seed. Two frozen same-batch probes check reproducibility before benchmark scoring.
The unlimited top-k setting is explicit. Unreported defaults cannot silently define the protocol.
The card's short example uses a 512-token cap. It does not explicitly set `do_sample` or `top_k`.
That example is not the paper's private reasoning evaluator.
This campaign matches the requested temperature and top-p. It does not claim to match unreported sampling filters.

Frozen inputs use the pinned tokenizer files through the existing Transformers 4.57 fast tokenizer.
The native engine consumes those token IDs without retokenizing the prompt.
Transformers 5 returns a different container and tokenizes repeated spaces differently in two secondary questions.
Those differences remain in the preflight audit. Every frozen ID sequence must decode to the exact official chat-template string.
No question, prompt string, or frozen input sequence changes after that discovery.

The paper uses an in-house harness and an LLM judge.
Our public deterministic evaluator is `REFERENCE_PROTOCOL_NOT_IDENTICAL`.
Its score is not automatically the paper's pass@1.
Problem-level bootstrap preserves all seeds from each sampled problem.
Technical repetitions and seeds are not independent problems.

No human judge changes an answer or score. The raw `manually_reviewed=False` field means no manual answer adjudication.
The assistant inspected two first-wave AIME outputs for EOS and truncation diagnostics only.
That inspection did not change prompts, selection, answers, grading, or the preregistered cap rule.

Primary sources:

- [Pinned model card](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/f1edd81e7ac41355db670500ceaf204e0f73af68/README.md)
- [Author project](https://ouro-llm.github.io/)
- [Paper, evaluation section and Appendix C](https://arxiv.org/html/2510.25741v2)
- [Official repository address](https://github.com/ByteDance/Ouro)

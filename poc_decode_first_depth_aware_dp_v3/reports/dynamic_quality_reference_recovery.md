# Stage 0R: source audit and baseline selection

The selection is `INCUMBENT_CONTINUE`. The incumbent already uses the CDB authors' implementation.
It uses the checkpoint's original gate. It does not use a separately trained gate.
The source SHA is `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
The public repository uses MIT licensing and lists third-party notices.

## Sources

| Candidate | Evidence | Decision |
|---|---|---|
| Official Ouro | Released weights and gate. The ordinary forward computes all recurrent steps before output selection. | Gate reference, not a physical-skipping serving replacement. |
| Official CDB | The incumbent imports this exact source. Its reproduction script uses min R2, online cumulative gating, and `last_exited`. | Same implementation, not an independent challenger. |
| Enabling Dynamic Computation | We checked the arXiv record, paper links, and exact-title repository searches on 2026-10-18. No author repository was verified. | `UPSTREAM_CODE_UNAVAILABLE` within this search scope. |
| Independent vllm-rlt | Public source SHA `9d601ceaf9af63121c742df5f584467e23630a0b`. | Audited. No port. This is not CDB authors' code. |

Source links: [Ouro](https://huggingface.co/ByteDance/Ouro-1.4B),
[CDB code](https://github.com/kschwethelm/continuous-depth-batching),
[Ragged-KV paper](https://arxiv.org/abs/2610.09013),
[vllm-rlt](https://github.com/ThinkFlowLab/vllm-rlt).

## KV semantics

The incumbent stores 96 depth-indexed K buffers and 96 V buffers.
At token exit, it copies that token's terminal KV rows into deeper slots.
The copy uses the compute stream. The copy precedes later recurrent reads.
Earlier depths retain their own KV values.
Thus a completed token supplies depth `min(current_depth, token_exit_depth)`.

This value rule matches the paper's best-available rule for completed tokens.
The storage mechanisms differ. The incumbent copies full slots.
The paper describes compact page aliases. This study does not claim the paper's memory savings.
The previous physical probe checked 360 exit-copy cases after synchronization.
The new B4 smoke executed 180 decode loop-token operations instead of 240 fixed-R4 operations.
It used the online original gate. All four sequences exited each decode token at R3 in that smoke.

## Preserved quality warning

The old v2 verdict remains `QUALITY_FAIL`.
The new scheduling study carries `QUALITY_WARNING` and `REFERENCE_MATCH_PARTIAL`.

| Domain | Fixed R4 | Adaptive | Fixed minus Adaptive | Paired question-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| GSM8K, 32 questions | 22/32, 68.75% | 21/32, 65.625% | 3.125 pp | [-6.25, 12.5] pp |
| MATH-500, 32 questions | 24/32, 75.00% | 19/32, 59.375% | 15.625 pp | [3.125, 28.125] pp |

These are inherited measurements, not new v3 results.
The primary cap was 4096. A paired truncation-union follow-up used 8192.
The v3 specification describes all incoming requests as 8192-cap runs. That description is not exact.
We retain the original artifacts. We perform a new 8192-cap adaptive recheck.
EOS-terminated old outputs remain valid continuity controls.

There is no independent, runnable, higher-quality challenger in this bounded audit.
The new recheck is not evidence of a separate algorithm's quality advantage.
It must not replace the old quality table silently.

## Protocol boundary

The shared operational metric uses free-running greedy generation and explicit answer extraction.
It uses the frozen 32 GSM8K and 32 MATH-500 question IDs.
GSM8K uses three CoT demonstrations. MATH uses a zero-shot boxed-answer prompt.
No new model, threshold, gate, or KV rule is selected from held-out outputs.

The Ragged-KV publication uses different prompts, rollouts, and a model judge.
Its reported percentages are not matched reproduction targets.
The CDB publication also separates quality evaluation from serving traces with recorded exits.
Our discovery trace uses the live gate, not synthetic exit replay.
[Ragged-KV evaluation](https://arxiv.org/html/2610.09013v1),
[CDB evaluation](https://arxiv.org/html/2608.09444v2).

## Continuation

The runtime is usable for Stage 1–3. The old quality warning does not stop those stages under v3.
No RL training, new gate training, or unrelated engine port is permitted.
New quality and scheduling results remain separate from these source-audit conclusions.

# Quality status

| Model | GSM8K | MATH-500 | Reason |
| --- | --- | --- | --- |
| Ouro R4 | NOT_RUN, N=0 | NOT_RUN, N=0 | No promoted adaptive method |
| Huginn R16 | NOT_RUN, N=0 | NOT_RUN, N=0 | No promoted adaptive method |

No missing accuracy receives a value of 0%.
The singleton and batched probes establish limited numerical fidelity, not benchmark accuracy.
The short arithmetic chats in DP portability are not GSM8K or MATH-500.
Their natural completion does not prove quality equivalence.

Any future candidate must reuse the preserved 64 GSM8K and 50 MATH pilot IDs.
Huginn requires its validated model-specific chat format and stop tokens.
The later evaluator must report completed-correct/N, caps, parser failures, and paired question-block intervals.
No GO claim can precede those tests.

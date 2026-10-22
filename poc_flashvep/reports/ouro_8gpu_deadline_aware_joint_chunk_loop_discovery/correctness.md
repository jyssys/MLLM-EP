# Correctness and quality scope

| Check | Result | Scope |
| --- | --- | --- |
| Isolated Atomic vs G1/G2/G3/staged-G4/Mixed | 160/160 exact | P128/512/2048/8192 across eight chunk sizes |
| Main prompt work | 1,728/1,728 cohorts pass | Four loop executions per recorded prompt token |
| Main decode work | 1,728/1,728 cohorts pass | Four loops per fed-back output token |
| Token clocks | 1,728/1,728 cohorts pass | Monotone clocks. Maximum alignment bound 0.166647 ms |
| Current legacy G2 vs Atomic | 56/64 exact | Separate from historical 72/80 |
| Current legacy Split vs Atomic | 56/64 exact | Same continuity lane |
| Current legacy Mixed vs Atomic | 37/64 exact | Same continuity lane |
| New quality G2 vs Atomic | 114/114 token-exact | Fixed-policy GSM8K/MATH pilot |
| New quality Mixed vs Atomic | 114/114 token-exact | Same pilot |
| Adaptive output/quality | NOT_RUN | No A1/A2/A3 live implementation |

The isolated probes compare hidden states, logits, tokens, and all 96 KV namespaces.
They do not certify every batched serving state.
The main work counts check execution accounting, not task accuracy.
The last sampled output token is not fed back into the model.

`first_divergence.md` documents one localized initial difference.
The captured MLP input and pretrained weights are identical.
Actual row counts M1024 and M48 reproduce the two BF16 outputs exactly.
Graph ON/OFF and stream controls preserve the captured result.
This case supports BF16 GEMM batch-geometry sensitivity.
It does not eliminate all lifecycle or state risks in other cases.

The validator-only G3 checkout also selected G3 rows with G1 remainder rows in one batch.
The executor rejected that mixed boundary.
The new guard defers spans with a different loop-group boundary.
The actual scheduler/executor regression is red before the guard and green after it.
`qualified_runtime_candidate.json` preserves the complete source difference.
No learned weight, precision, or decode depth changed.

Read `quality_summary.md` for correct/N, complete-correct counts, caps, and paired intervals.
The new pilot scores GSM8K 47/64 and MATH 28/50.
MATH has eight capped answers, including two extracted-correct answers.
All three fixed-policy outputs match in this pilot.
The 300+100 and full 1,819-question evaluations remain NOT_RUN.

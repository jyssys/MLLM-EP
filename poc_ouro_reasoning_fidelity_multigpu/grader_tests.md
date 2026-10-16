# Public grader checks

The AIME parser accepts a final integer from 0 through 999.
It uses the last balanced boxed expression when one exists.
It otherwise requires an explicit final-answer marker.
An arbitrary integer in reasoning does not become the final answer.
The parser does not execute generated code.

The CPU unit suite checks seven extraction cases.
It checks a wrong answer, an absent answer, and an out-of-range answer.
It also checks that a matching answer inside a capped output cannot count as completed correct.
The denominator test has one completed correct output among three outputs.
Its score is 1/3, not 1/2 or 2/3.

MATH uses the existing strict Math-Verify 0.8.0 path.
The previous frozen 100-answer parser audit remains provenance, not new benchmark accuracy.
The final scoring command must use this campaign's raw outputs and a new output directory.
It must not overwrite the preceding experiment's score files.

All score changes require a recorded evaluator cause.
No manual adjudication, desired-score correction, or paid judge is authorized.
Problem-level bootstrap preserves all seeds within each problem.
Three sampled outputs do not support pass@10.

Evidence: `results/grader_unit_tests.json` and `tests/test_contract.py`.
The final audit records the actual test command and result.

## Completed-output audit

All 91 naturally terminated MATH100 outputs have a parsed answer.
The unchanged strict grader rejects four completed answers.
Two outputs give one branch rather than the reference's plus/minus expression.
This descriptive audit changes no grade.

| Frozen problem | Extracted answer | Reference |
|---|---|---|
| test/geometry/172.json | 16 | 8 |
| test/intermediate_algebra/662.json | `1 - sqrt(19)` | `1 +/- sqrt(19)` |
| test/precalculus/768.json | `3 - 2sqrt(2)` | `3 +/- 2sqrt(2)` |
| test/intermediate_algebra/1544.json | 4 | 6 |

The AIME completed-unparsed ledger contains four records across all caps.
They represent three problems, not four independent problems.
The extracted expressions are 3268, 1024 twice, and `199 - sqrt(8481)`.
None is a valid integer from 0 through 999.
The audit finds no valid AIME final answer hidden by those four normalization failures.

Evidence: `results/completed_unparsed_answer_ledger.json` and `results/quality_score/quality_generations.csv`.

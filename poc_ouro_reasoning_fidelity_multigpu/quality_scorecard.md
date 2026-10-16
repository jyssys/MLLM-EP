# Public quality scorecard

`REFERENCE_PROTOCOL_NOT_IDENTICAL` applies to every score below.
The public grader differs from the authors' in-house harness and LLM judge.
The checkpoint, P1 prompt, R4, BF16, and seeds remain frozen.
The primary sampled policy uses temperature 1.0, top-p 0.7, and no top-k limit.
Greedy results use temperature 0 and are a separate baseline.

## Complete cohorts

| Dataset / policy | Cap | Completed correct / all | Problem-bootstrap 95% CI | EOS / all | Capped | Parsed / all |
|---|---:|---:|---:|---:|---:|---:|
| aime24 / greedy | 8,192 | 13/30 = 43.33% | 26.67%–60.00% | 13/30 | 17 | 14/30 |
| aime24 / greedy | 16,384 | 16/30 = 53.33% | 36.67%–70.00% | 17/30 | 13 | 17/30 |
| aime24 / greedy | 24,576 | 19/30 = 63.33% | 46.67%–80.00% | 21/30 | 9 | 21/30 |
| aime24 / sampling | 8,192 | 37/90 = 41.11% | 25.56%–56.67% | 41/90 | 49 | 44/90 |
| aime24 / sampling | 16,384 | 51/90 = 56.67% | 41.11%–72.22% | 61/90 | 29 | 64/90 |
| aime24 / sampling | 24,576 | 52/90 = 57.78% | 43.33%–72.22% | 79/90 | 11 | 78/90 |
| math100 / greedy | 8,192 | 87/100 = 87.00% | 80.00%–93.00% | 91/100 | 9 | 94/100 |

All outputs remain in the denominator, including length stops.
A matching extracted answer inside a length-stopped output does not count as completed correct.
The bootstrap uses 5,000 problem-cluster resamples.
All seeds for one sampled AIME problem remain together.
There are 30 independent AIME problems, not 90 independent problems.
MATH100 is the frozen 100-problem subset, not the full MATH-500 benchmark.

## Final declared caps and lengths

| Dataset / policy | Final cap | Generated tokens P50 / P90 / P95 / max | Mean repeated 16-gram fraction |
|---|---:|---:|---:|
| aime24 / greedy | 24,576 | 9,230.0 / 24,576.0 / 24,576.0 / 24,576.0 | 0.18756 |
| aime24 / sampling | 24,576 | 10,129.5 / 24,576.0 / 24,576.0 / 24,576.0 | 0.07400 |
| math100 / greedy | 8,192 | 1,838.5 / 6,355.8 / 8,192.0 / 8,192.0 | 0.03873 |

## Seed-wise sampled AIME result

| Seed | Completed correct / all |
|---|---:|
| 41015 | 19/30 |
| 41016 | 16/30 |
| 41017 | 17/30 |

Pass@10: `NOT_RUN`.
Three seeds cannot estimate the authors' ten-sample pass@10 protocol.

## Cap and request-history sensitivity

| Dataset / policy | Cap transition | Changed common prefix / all | Paired completed-correct difference, pp | Problem-bootstrap 95% CI, pp |
|---|---:|---:|---:|---:|
| aime24 / greedy | 8,192 → 16,384 | 7/30 | +10.00 | +0.00–+20.00 |
| aime24 / greedy | 16,384 → 24,576 | 3/30 | +10.00 | +0.00–+23.33 |
| aime24 / sampling | 8,192 → 16,384 | 20/90 | +15.56 | +7.78–+24.44 |
| aime24 / sampling | 16,384 → 24,576 | 20/90 | +1.11 | -4.44–+7.78 |

The cap comparison also changes the engine's preceding request history.
Changed common prefixes prevent a censoring-only explanation of score differences.
Fixed per-request seeds do not remove this limitation.
The separate fixed-history diagnostic does not replace any public benchmark output.

## Author comparison

| Property | Author reference | This public evaluation |
|---|---|---|
| AIME2024 R4 pass@1 | 64.7% | See the sampled final-cap row above |
| AIME2024 R4 pass@10 | 90.0% | NOT_RUN: three seeds only |
| Sampling | Temperature 1.0, top-p 0.7 | Temperature 1.0, top-p 0.7, explicit unlimited top-k |
| Harness and grading | Private harness and LLM judge | Frozen public integer exact match |
| Exact AIME cap, prompt, seed stream | Not fully disclosed | Frozen public protocol and raw token IDs |

The source is [Ouro Table 9 and Appendix C](https://arxiv.org/html/2510.25741v2).
The paper's separate 8,192-token safety experiment does not establish its AIME cap.
A confidence interval that includes 64.7% does not establish protocol equivalence.
The campaign does not select a protocol by proximity to the published score.

## Grading and stop checks

Every observed natural stop ends with EOS ID2.
No length-stopped output contains EOS ID2.
No recorded generation exceeds its effective cap or the 32,768-position limit.

The unchanged grader records completed but unparsed outputs separately.
The ledger includes invalid answer formats and out-of-range AIME integers.
It does not automatically classify every unparsed answer as a grader defect.
No manual grade, EOS rule, or benchmark answer changes.

Operational quality, reference fidelity, and systems authorization remain separate decisions.
See `results/quality_decision.json` for the operational decision.

Evidence: `results/quality_score/quality_generations.csv`, `results/quality_cap_analysis.json`, and the raw-output archive.

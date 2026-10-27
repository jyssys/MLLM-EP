# Interim checkpoint: completed evidence

Status: **IN_PROGRESS**. This commit is not the final PoC release.

Snapshot date: 2026-10-27, Korea Standard Time.
The active GPU campaigns continue during publication.
The final prototype verdict remains **PENDING**.

## Completed measurements

| Evidence | Completed scope | Result or interpretation |
|---|---|---|
| TRUE EP hardware matrix | 90/90 cells | Three models, EP4/EP8, B1/2/4/8/16, prefixes 128/512/2,048 |
| Matched native hardware controls | 45/45 cells | Actual official checkpoints, not proxy models |
| Native versus EP generation | 90/90 cells | All recorded first-repetition 32-token greedy outputs match |
| Available native official quality | Seven full task evaluations | Published levels reproduced within the declared compatibility margin |
| Available EP official quality | Fourteen full task evaluations | Benchmark correct counts match the native configurations |
| Foil held-out NLL | Native, EP4, EP8 | 2.4493475213060765 nat/token in all three configurations |
| Independent-start quiet controls | 12/12 attempts | Completed raw records, separate from the primary serial-repetition matrix |
| Recurrent affinity analysis | All three original models | Held-out transition prediction improves over pass/domain popularity |

The native comparison has 180 rows because each EP shape includes two placement policies.
Native hardware controls use the full 15-shape matrix for each model.
IFM B16/prefix-2,048 native recovery uses an experimental allocator setting.
The affected native-versus-EP rows explicitly report this allocator difference.

These adapters recompute the full prefix for each generated token.
They do not retain a KV cache in the measured path.
The 32-token hardware rollouts disable EOS.
They are not natural client-to-server serving measurements.

## Official quality anchors

| Model | Benchmark | Native and EP4/EP8 correct/N | Accuracy | Published reference |
|---|---|---|---:|---:|
| IFM Ouro | MMLU, five-shot micro accuracy | 8,979/14,042 | 63.9439% | 64.03% |
| IFM Ouro | ARC-Challenge, official zero-shot acc_token | 599/1,165 | 51.4163% | 51.33% |
| IFM Huginn | MMLU, five-shot micro accuracy | 9,379/14,042 | 66.7925% | 66.81% |
| IFM Huginn | ARC-Challenge, official zero-shot acc_token | 592/1,165 | 50.8155% | 50.47% |
| Foil-1 | LAMBADA standard, zero-shot | 1,632/5,153 | 31.6709% | 31.83% |
| Foil-1 | LAMBADA OpenAI, zero-shot | 2,056/5,153 | 39.8991% | 39.98% |
| Foil-1 | XWinograd English, zero-shot | 1,603/2,325 | 68.9462% | 68.82% |

See [accuracy_reproduction.md](accuracy_reproduction.md) for Wilson intervals, paired intervals, and protocol limitations.
The compatibility rule is the larger of 2 percentage points and the Wilson interval half-width.
It checks approximate reference reproduction, not statistical equivalence.
The paper Foil held-out reference is 2.4492 nat/token.
The measured loss uses 7,997,535 shifted targets in 1,953 complete 4,096-token windows.
The corpus recipe matches, but the paper does not publish its original binary token hash.

HellaSwag remains unavailable from the official distribution.
The complete eleven-task IFM paper suite is not reproduced.
GSM8K, MATH-500, and DROP receive no invented accuracy result.
Foil uses different library versions from the published requirements.

The current LAMBADA paired table retains 5,151 unique document hashes from 5,153 evaluated items.
The full aggregate scores cover all 5,153 items.
Duplicate-item pairing still needs a separate correction before a full item-level agreement claim.

## Promising evidence and negative controls

| Model | Lag-1 cells | Conditional Recall@2 | Popularity Recall@2 | Increment |
|---|---:|---:|---:|---:|
| IFM Ouro | 5,040 | 0.55472 | 0.45660 | 0.09812 |
| IFM Huginn | 5,040 | 0.55023 | 0.42390 | 0.12633 |
| Foil-1 | 900 | 0.42799 | 0.20367 | 0.22432 |

These descriptive means weight analysis cells equally.
They are not pooled independent-prompt estimates or measured speedups.
Per-cell intervals preserve the prompt as the resampling unit.

No loop-specific E2E speedup of at least 1.15× is established.
At Ouro EP4, B1, prefix 128, the generic source-owner control reduces remote dispatch payload by 46.397%.
Its static-to-control E2E speedup is 0.977771×.
Lower payload therefore does not establish lower latency.

The placement trace analysis also trades lower fanout against higher rank-load imbalance.
Recurrent maps add little over the tested generic maps in several conditions.
Those maps use finite heuristics, not an exact placement optimum.
They do not establish a wall-time upper bound.

## Active experiments and their decision value

1. The strong generic metadata control tests a redundant count All-Gather in the existing bridge.
2. The whole-E2E observer measures exclusive component milliseconds and percentages across the complete rollout.
3. The final analysis compares legal recurrent mechanisms against these generic costs and controls.

The generic control derives counts from exact current router outputs that already exist on every rank.
It uses neither future routing nor recurrent prediction.
Its gain, if any, is generic runtime headroom, not recurrent novelty.

The E2E observer must pass its overhead and alignment gates.
Summed kernel service times cannot substitute for additive E2E component percentages.
No pending percentage is filled with an estimate.

These experiments are necessary to interpret the observed bottleneck.
They cannot prove a new loop-specific architecture merely by completing successfully.
The strongest production generic EP comparator and a validated placement wall-time model remain missing.

## Code, raw evidence, and release status

- [report.md](report.md) and [summary.md](summary.md) remain explicit previews.
- [benchmark_summary.md](benchmark_summary.md) contains the completed 90-cell hardware matrix.
- [native_oom_diagnosis.md](native_oom_diagnosis.md) preserves the actual failure and allocator-only recovery.
- [quality/checkpoint_evidence_manifest.json](quality/checkpoint_evidence_manifest.json) identifies closed numeric evidence and campaign states.
- [archives/checkpoint_completed_numeric.jsonl.gz](archives/checkpoint_completed_numeric.jsonl.gz) preserves the selected completed numeric records.
- Complete raw evidence remains local under `raw/`. This publication does not delete it.

Pending attribution and metadata-control code is experimental infrastructure.
Its presence in this commit does not certify an unmeasured result.
All fourteen CPU regression tests pass.
These tests include checkpoint publication, final publication, and preservation of unrelated staged files in isolated Git fixtures.
The checkpoint evidence test rejects active attempts from the completed numeric archive.
CPU tests do not establish GPU timing validity or downstream model quality.
The final publisher accepts this checkpoint only after its push and remote SHA verification.
Unrelated staged work remains outside this study's commit scope.

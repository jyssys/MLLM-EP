# Atomic Prefill PD: Correctness Evidence

Status: qualification checkpoint, not final lossless-model certification.
All results below come from executed GPU runs.
Performance and sustainable capacity are separate questions.

## Long-context Looped gates

| Model | Input/output | Full-Prompt spans | Local-reference output parity | Valid KV and final hidden byte identity | Active recurrent rows, expected/observed | Result |
|---|---:|---:|---:|---:|---:|---|
| Ouro R4 | 8192/16 | 8/8 | 8/8 | 8/8 | 262624/262624 | FINITE_GATE_PASS |
| Huginn R32 | 3900/16 | 8/8 | 8/8 | 8/8 | 1002240/1002240 | FINITE_GATE_PASS |

Evidence type: MEASURED_HEAVY_CORRECTNESS.
Each row uses eight unique test prompts.
These probes include large device-to-host hashes.
Their latency is not clean Serving latency.

Ouro uses `qual_ouro_full_l8192_v1` and `qual_ouro_local_l8192_v2`.
Huginn uses `qual_huginn_full_l3900_v3` and `qual_huginn_local_l3900_v3`.
The state records distinguish valid recurrent KV from unused storage planes.
Huginn's final Prompt Coda KV is not valid before that Coda executes.
The comparison preserves that validity boundary.

The work check counts active recurrent rows.
It does not count padded GEMM rows as real model tokens.
The numerical fixes retain fixed 256-row Down tiles.
Huginn also retains its qualified adapter geometry.

## Token-chunk controls

| Model | Input | Candidate C | Output parity against Atomic Full-Prompt | Full-depth accounting |
|---|---:|---|---|---|
| Ouro R4 | 8192 | 256, 1024, 2048, 4096 | 8/8 for each C | PASS for each C |
| Huginn R32 | 3900 | 256, 1024, 2048 | 8/8 for each C | PASS for each C |

Evidence type: MEASURED_OUTPUT_AND_WORK_COUNTERS.
C4096 and Full-Prompt are the same effective chunk at Huginn input 3900.
They are not two independent mechanisms.
The older staged-G1 C2048 control also matches the new Full-Prompt output in 8/8 requests for both models.
That comparison does not establish equal scheduling overhead.

## Native AR evidence

Qwen3-8B loads the exact pinned 8.19073536B-parameter checkpoint.
Each PD worker holds 16,381,470,720 bytes of BF16 parameters.
Native NIXL produces actual remote KV descriptors and transfer telemetry.
No TP, trace replay, or smaller model replaces the requested AR model.

| AR setting | Comparison | Exact sequences | Interpretation |
|---|---|---:|---|
| Default native PD, 8000/16 | First start vs first repeat | 8/8 | Finite within-PD repeat pass |
| Default native PD, 8000/16 | Third start vs both earlier starts | 8/8 against each | Three matching starts on the same eight prompts |
| Default native PD vs first default DP2 | Matching prompts and seed | 7/8 | Failed cross-mode equality |
| Optional native batch-invariant PD vs DP2 | Matching prompts and seed | 7/8 | Native flag does not establish equality |
| Optional batch-invariant PD repeat | Matching native-PD configuration | 7/8 | Do not expand this optional mode |
| Default DP2 repeat | Earlier and later colocated starts | 7/8 | Colocated nondeterminism also occurs without NIXL |

The first recorded divergent case is sample 6.
Its differing continuation includes token ID 1033 in one run.
The optional-mode native-PD repeats first differ at output index 7.
Raw input IDs, seeds, output IDs, model revision, and source archives remain in each namespace.

The no-NIXL counterexample shows that NIXL is not necessary for every observed AR mismatch.
It does not prove the cause of the native-PD optional-mode mismatch.
No internal first-divergent tensor or logit margin has been measured.
Do not describe BF16 rounding as an established diagnosis.

The original default native-PD family can proceed only with matching candidate-chunk gates.
Default C256 and C1024 match 8/8 at 8000/16.
Default C2048 and C4096 each fail on sample 6.
Those failed C settings cannot become clean performance baselines.
A later colocated match does not erase the earlier failed colocated repeat.
Exact lossless PD-versus-DP2 fidelity remains unproven.

## Failed compatibility runs

`qual_ouro_local_l8192_v1` failed because its local harness did not enable existing work counters.
The missing field was `issued_loop_rows`.
The correction sets `ABC_RUN_DIR` before engine construction.
It does not change a model operation or an acceptance threshold.
The fresh `v2` namespace passes the local reference check.

The first Huginn 3900-token attempts failed with CUDA OOM.
The retry reduces the common KV pool from 26,624 to 24,576 blocks.
It also limits the aggregate P token budget to 7,800.
The new P/D reserved KV is 66,437,775,360 bytes per GPU.
Observed qualification peaks are 76,377,750,528 bytes on P and 74,789,390,336 bytes on D.
These are memory-compatibility measurements, not clean latency gains.

## Required later checks

The controller must run an independent 32-request gate before a long final candidate campaign.
That gate uses another arrival seed, another sampling seed, and Prompt pool rows 32–63.
Full-Prompt native PD must repeat exactly.
Each tuned or colocated candidate must match its eligible reference.
Failed settings remain blocked.

Live final audits compare exact output prefixes within a matched runtime family and context geometry.
They also check active full-depth work and actual Atomic spans.
The Looped runtime family keeps its strict cross-method checks.
Known AR cross-mode failures remain separate explicit limitations.

## Scope limits

Original-HF fidelity: NOT_MEASURED in this PoC.
Downstream answer accuracy: NOT_MEASURED in this PoC.
Natural-EOS generation quality: NOT_MEASURED in this PoC.
An eight-request gate is not a universal lossless proof.
All main outputs are controlled-cap greedy generation with EOS ignored.
Final qualification and Serving results must be read with their actual request counts and failure status.

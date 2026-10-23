# Full-answer quality pilot

Status: COMPLETED public pilot. No population-equivalence or universal token-parity PASS is declared.

## Frozen protocol

The dataset contains 64 GSM8K test questions and 50 MATH-500 questions per arm.
The nine arms therefore register1,026 free-running generations on 114 unique questions.
These public questions also appear in earlier studies. They are not a fresh blind accuracy set.
The complete prompts, input IDs, gold answers, and registered stop strings remain in `quality_manifest.json`.
The manifest records its preserved source and SHA256.

The model is Base Ouro-1.4B at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Every arm uses BF16, greedy temperature0, maximum recurrence4, and exit threshold 1.
Every prompt and generated token must execute Full R4.
The output cap is 4,096 tokens. Native EOS and preserved frontend stop rules remain active.
No frozen output replay or teacher forcing contributes to accuracy.
The short256-cap callback smoke is excluded from all accuracy tables.

Four workers partition question indices modulo4.
The same question therefore crosses every policy on the same physical GPU.
Policy order rotates among GPU4/5/6/7 model-process blocks.
Batch width is 4. Per-GPU case counts are 29/29/28/28 per arm.
The engine uses native graphs, streams, KV namespaces, and sampling.
The work ledger checks prompt-token loop work against4×input length at each engine teardown.
Generated exit-depth traces must contain only 4.

## Preserved evaluator

The evaluator imports `poc_looped_exact_depth_chunk_pipeline/scoring.py` from the older preserved worktree.
Its actual imported file SHA256 is checked against the inherited extractor lock.
The standalone import check resolves:
`/home/esjung/MLLM-EP-ouro-request-pp/poc_looped_exact_depth_chunk_pipeline/scoring.py`.
SHA256: `ddb94ddc472c4bc400d4f241861e170d21037b9da6e6a8d5c1ee2c25efea1c78`.
The scorer's existing parsing and answer-equivalence rules remain unchanged.
Parser-only API fixtures are software checks, not model accuracy samples.

Extracted-correct counts any preserved-evaluator answer match.
Complete-correct additionally requires a true registered stop rather than the output cap.
Both denominators retain every offered benchmark question.
Parse failures, caps, output lengths, and baseline-correct-to-candidate-wrong cases remain explicit.
The cap does not become an accuracy-speed trade-off.

## Paired statistics

The paired bootstrap resamples questions within each dataset, with 5,000 draws.
The frozen RNG seed is 2026102303.
It never treats generated tokens or repeated policy outputs as independent quality samples.
Comparisons use stock Atomic F0A and direct same-objective G/C controls.
Differences use percentage points, not relative percentage change.
The registered adverse point margins are 1.0pp GSM8K and 2.0pp MATH.
Confidence intervals and the small sample size remain separate from that point screen.
Zero observed difference does not prove population equivalence or universal output fidelity.

The quality client decodes text and checks stop strings during execution.
Its timings are not clean serving latencies or qualified capacity measurements.
Independent numerical probes and natural token parity also remain separate evidence classes.

## Measured results

All four jobs complete1,026 registered generations.
All 36 prompt ledgers pass Full-R4 accounting. Every generated exit-depth trace contains only 4.
The table reports actual correct/total counts, not an estimate from activation similarity.
GSM8K complete-correct equals extracted-correct in every arm. No GSM8K answer is capped.

| Arm | GSM8K correct /64 (%) | GSM8K parse failures | MATH extracted correct /50 (%) | MATH complete-correct /50 (%) | MATH parse failures | MATH caps |
| --- | --- | ---: | --- | --- | ---: | ---: |
| F0A stock Atomic |47/64 (73.44) |6 |29/50 (58.00) |25/50 (50.00) |4 |8 |
| `BEST_FIXED` (C512/G4) |47/64 (73.44) |6 |29/50 (58.00) |25/50 (50.00) |4 |8 |
| Stock Mixed |47/64 (73.44) |6 |29/50 (58.00) |25/50 (50.00) |4 |8 |
| Completion-fair Mixed |47/64 (73.44) |6 |29/50 (58.00) |25/50 (50.00) |4 |8 |
| O6 C-only, staged |47/64 (73.44) |6 |32/50 (64.00) |29/50 (58.00) |3 |7 |
| O6 C-only, Atomic |47/64 (73.44) |5 |32/50 (64.00) |29/50 (58.00) |3 |6 |
| O6 G-only C2048 |47/64 (73.44) |6 |32/50 (64.00) |30/50 (60.00) |3 |6 |
| O6 G-only C1024 |47/64 (73.44) |6 |32/50 (64.00) |30/50 (60.00) |3 |6 |
| O6 Joint |47/64 (73.44) |6 |32/50 (64.00) |30/50 (60.00) |3 |6 |

## Paired accuracy and token agreement

Joint and both G-only controls have identical correctness labels on all 114 questions.
Their measured extracted-accuracy difference is 0pp in each domain.
The question-paired bootstrap interval is[0,0]pp because this observed set has no correctness discordance.
That degenerate interval describes this set. It is not proof of population noninferiority at the 1pp/2pp margins.
The small historical public pilot cannot exclude rare regressions on unseen requests.

| Joint comparison | Dataset | Accuracy difference pp | Paired95% interval pp | Token-exact /N | Correct→wrong / wrong→correct |
| --- | --- | ---: | --- | --- | --- |
| G2048 | GSM8K |0.00 |[0,0] |64/64 |0/0 |
| G1024 | GSM8K |0.00 |[0,0] |64/64 |0/0 |
| G2048 | MATH |0.00 |[0,0] |48/50 |0/0 |
| G1024 | MATH |0.00 |[0,0] |48/50 |0/0 |
| Staged C-only | MATH |0.00 |[-8,8] |33/50 |2/2 |
| Atomic C-only | MATH |0.00 |[-6,6] |25/50 |1/1 |
| Stock Atomic F0A | GSM8K |0.00 |[0,0] |62/64 |0/0 |
| Stock Atomic F0A | MATH |6.00 |[-4,16] |22/50 |2/5 |

Joint/G2048 token agreement is 112/114 overall, not 114/114.
The two MATH token mismatches have equal scored outcomes and equal stop reasons.
They still prevent a universal token-exact claim.
Against stock Atomic, Joint loses two previously correct MATH questions and gains five others.
The net6pp point increase is uncertain and is also present in G-only.
It cannot establish a C-specific quality gain or justify a speed claim.

The earlier C-only 43/64 GSM8K result remains preserved under its original selector/batch protocol.
It is not reproduced as a new drop by the current O6 C-only pilot, which scores47/64.
These are different configurations and grouping histories. The new result does not erase the earlier regression.
The underlying source of all batched numerical/output differences remains unresolved.
The 32 isolated numerical controls do not replace whole-batch or whole-generation fidelity evidence.

## Quality decision and limits

The observed same-objective Joint/G-only quality comparison passes the pilot's adverse point screen.
The broader quality status remains CONDITIONAL because of sample size, caps, and incomplete token parity.
No full1,819-question evaluation runs after the incremental serving gate fails.
No official model-paper accuracy reproduction is claimed for this Base-model scheduling study.
SLO goodput on Dolly/arXiv requests is not a per-answer correctness metric.

Data: `quality_summary.csv`, `quality_cases.csv`, `quality_paired_comparisons.csv`, and `quality_matched_controls.csv`.
Regression examples: `quality_baseline_correct_to_wrong.csv` and preserved raw outputs.
Work checks: `quality_prefill_ledger_validation.csv` and `quality_analysis_status.json`.
Portable full outputs appear in `portable_quality_outputs.jsonl.gz` after evidence export.

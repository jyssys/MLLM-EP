# Progress and resume notes

P0 short native-versus-EP validation completed for all three models at EP4 and EP8.
All six initial comparisons have zero relative logit error and exact greedy smoke outputs.
This establishes the thin bridge at the tested input, not paper-quality reproduction.

The complete B1/B2/B4 baseline campaign is running.
Its saved plan defines process-start and workload coverage.
The final quality campaign adds long-shape checks and 32 engineering examples per domain.
The P1 ready observer waits for those campaigns before it uses GPUs.
No optimization verdict is available from these notes alone.

The waiting CPU quality watcher was replaced before it launched GPU work.
The revised watcher includes the full shape and corpus checks.
No running GPU experiment was stopped for that change.

The code preserves native IFM and Foil source files.
Each worker saves its source hashes before loading.
The dispatcher skips complete run IDs and refuses to overwrite partial attempts.
Owned burners remain on unused GPUs.

## Completed experiment checkpoint

All 42 jobs completed successfully.
The evidence contains 252 distributed rank executions.
The final engineering corpus passes 105/105 checks in each model/EP cell.
The CPU evidence audit passes 2,679 checks.

The clean matrix contains 54 model/degree/batch/prefix cells.
The ready observer covers 36 independent-request arrival cases.
The warmed generation control covers all six model/EP configurations.
Foil's first full generation is approximately 1.90× to 2.02× its final warmed repetition.
The final two warmed repetitions remain within 10% in all six configurations.
This separates shape-preparation drift from a measured method speedup.

Track A gives 1.000× against its best generic wait window in the tested models.
Track B adds less than 1% over equal-budget generic replication.
Those values are modeled estimates, not actual candidate E2E acceleration.
Neither track clears the research gate.
The experiment therefore stops before optimized A/B dispatch implementation.

All 43 raw/log archives retain original evidence and verified SHA-256 hashes.
The compressed evidence totals 149,178,320 bytes.
Original local raw files remain unchanged.
The publication step must preserve unrelated staged entries.

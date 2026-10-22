# Offline joint opportunity and its limits

Phase 4 status: **BLOCKED for a qualified scheduling-headroom conclusion**.
`joint_headroom.csv` contains 88 modeled execution-budget cells.
It does not contain live adaptive results.

The analysis constructs 70 legal single-chunk candidates from measured TRAIN/CALIBRATION costs.
Candidates retain full R4, exact chunk ownership, clipped remaining depth, and one staged chunk.
The initial quantum includes staged prelude cost where applicable.
Later loop groups contribute to an estimated full-chunk cost.
C-only can choose every supported full-size C with G4.
A stronger control also permits the staged-G4 path.
G-only fixes C1024.
Joint can choose supported C/G combinations.

At P≈8K/N8 and a 100-ms initial execution budget, the heuristic selects:

| Menu | Selected action | Estimated full-chunk rate | Immediate completed prompt tokens |
| --- | --- | --- | --- |
| Atomic C-only | C768/G4 | 7.753 tokens/ms | 768 |
| Best G4 paths | Staged C768/G4 | 8.030 tokens/ms | 768 |
| G-only C1024 | C1024/G2 | 9.655 tokens/ms | 0 |
| Joint | C2048/G2 | 15.074 tokens/ms | 0 |

This is evidence of a possible execution-granularity trade-off.
It is not a 1.88× serving speedup or a proven deadline advantage.
Partial token-loop work is not completed prompt progress.

## Explicit missing costs and state

The envelope uses coarse context buckets and median costs.
It excludes clipped final chunks and mixed/decode co-batches.
It does not model later decode interruptions, queue rollout, fairness changes, or complete-request TTFT.
Some measured costs include overlapping waits.
Their sum is a heuristic estimate, not an exact isolated service decomposition.
The envelope is not a formal optimistic bound.

Native Mixed cannot be fairly inserted into this single-chunk score.
It packs decode rows and prompt rows together.
Actual Mixed serving points remain in `cost_surface_summary.csv` and the Pareto figure.
This analysis does not establish Joint superiority to Mixed.

Qualified opportunity frequency in real held-out states is NOT_AVAILABLE.
The captured logical frontier does not fully identify physically published state before selection.
The D1 prediction describes the stock future schedule, not each alternate action.
The prefill-specific safety bound also misses the registered coverage target.

`legal_actions.py` and its tests define legal action and bounded decode-DAG rules.
Their tests are software checks, not latency validation.
No unvalidated trace replay is presented as a causal E2E speedup.

## Next gate

Do not implement A3 from this envelope alone.
First collect pre-selection readiness events and validate a candidate-specific complete-token model.
Then test actual state frequency and compare tuned C-only and Mixed under the same information.
Final online source families remain untouched.

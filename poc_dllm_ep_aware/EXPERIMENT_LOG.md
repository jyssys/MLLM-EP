# Experiment log

All new GPU launches used physical GPUs 0–3 only.  GPU burn is excluded from
research time and all metrics.

## 2026-09-12 02:55–03:17 KST — TEAM EP4

- Verified 128 experts with contiguous 32-expert ownership per rank.
- Validated rank-0 EP layer against the unsharded layer: cosine 0.99999994,
  relative L2 0.0560%.
- Collected randomized clean baseline/TEAM pairs and observer-heavy work trace.
- One TEAM attempt hung after its paired baseline; terminated the task-owned
  process group and excluded the sample.
- Observed TEAM median 0.821× speedup at EP4 and +32.1% remote assignments.

## 2026-09-12 03:18–03:49 KST — LLaDA policy transfer

- Corrected the LLaDA mask token to 156895; all wrong-mask artifacts excluded.
- Tested vanilla, paper-faithful REFLEX RRB/FGER, and DES-Vote active-block
  coreset on single GPU, EP2, and EP4 reference assignment A2A.
- REFLEX and DES failed the bounded GSM8K quality gate.  Preserved their traces
  only for action-to-physical-cost mapping.
- Confirmed selected-pair reduction does not reduce bytes in stock naive EP;
  assignment-aware reference transport does reduce rows but not fanout.

## 2026-09-12 04:00–04:01 KST — EP4 cost surface

- Swept 84 route shapes over assignments, fanout, remote fraction, and active
  experts/rank with 30 measured repetitions after five warmups.
- Found a dominant approximately 0.61 ms startup floor; 16× assignments changed
  median operator time by only about 4%.
- Held-out critical-rank model was best but non-explanatory (R²=0.058).

## 2026-09-12 04:04–04:06 KST — clean replication

- Added the third successful TEAM EP4 paired restart.
- Final per-restart TEAM speedups: 0.902×, 0.716×, 0.821×.

## Oracle and method decision

- Evaluated candidates A–F with request-level upper bounds.
- Best was perfect-future Candidate F at 6.28%; best implementable candidate E
  was 4.01%.
- Both fail the implementation gate.  No live method prototype was built.

# Same-prefix native numerical control — 2026-09-09

Result directory: `nanoflow_runs/numerical_resume_20260909_b4_retry1` under the
existing study results root. Four fresh native engines: two graph-unsplit and
two graph-split2 runs, randomized order, the same four real-content requests,
128-token contexts and 32 teacher-forced output steps. No numerical kernel or
router modification. Rank-zero native vocabulary logits captured after execute.

## Observed

- All four worker groups exit successfully on physical 4–7.
- 128 token positions in each comparison: graph restart, split restart and both
  cross-plan restart pairs each have **100% argmax agreement**.
- Each run also matches all 128 original unsplit reference tokens at the same
  teacher-forced histories. This is not free-generation benchmark accuracy.
- Logits are not bitwise identical even between same-plan restarts. Mean KL:
  unsplit restart 1.02e-4, split restart 7.89e-5, cross-plan pairs 1.13e-4 and
  8.67e-5. Cross-plan numerical differences are not clearly larger in this sample.
- The earlier free-generation split mismatch is therefore not established as a
  deterministic split-semantic failure. Larger cohorts and free-generation
  restart controls remain mandatory; four requests do not clear all workloads.

All these runs are **correctness diagnostics only**. Forced histories and logit
D2H overhead make their times ineligible for performance/oracle claims.

CPU analysis: `analyze_numerical_screen.py`; artifacts `same_prefix_logits.csv`
and `same_prefix_summary.json`. Stage/capture work remains separate from E2E.
# Batch-16 extension — 2026-09-09

Four fresh workers groups, same 16 real-content prompts (512 input tokens) and
32 identical teacher-forced output histories, completed on physical 4–7.
Each comparison has 512 token observations. Within-unsplit restart argmax
agreement is 100%; within-split is 511/512. Cross-plan agreements are 510/512 and
509/512. All cross-plan mismatches have unsplit top-two margin 0–0.015625;
mean KL is 0.00005188 / 0.00007718. Same-split restart mean KL is 0.00008130.
The historical unsplit reference also differs slightly from newly restarted
unsplit runs, so that historical output is not an infallible bitwise oracle.

This weakens a deterministic graph-split semantic bug explanation, but is not
a task-quality equivalence proof. Do not erase mismatches or treat teacher-forced
timing as free-generation performance. Keep high-margin mismatch checks and
independent task correctness distinct from exact text hashes.

Artifacts: `nanoflow_runs/numerical_resume_20260909_b16/`:
`same_prefix_logits.csv`, `same_prefix_summary.json`, per-step raw logits.

## Independent HF same-prefix check — 2026-09-09 11:30 KST

`nanoflow_runs/hf_numerical_reference_20260909/` contains an independent FP16
HuggingFace/SDPA forward on the same checkpoint and exact recorded token history.
Native execution uses incremental FlashInfer KV; HF recomputes the causal prefix.
No timing from this validation is performance evidence.

Across all four B4 engines, HF argmax agrees at 127/128 positions; the common
mismatch has HF top-two gap zero. B16 agreement is 508/512 in three engines and
507/512 in split2 restart 1. All HF-disagreeing positions have reference top-two
gap <=0.015625. Mean HF-to-native KL ranges approximately 6.05e-5–1.01e-4.
This rejects neither near-tie numerical sensitivity nor downstream free-generation
drift; it does weaken an obvious high-margin semantic split failure. There is
still no task benchmark equivalence claim.

## M8192 prefill resource control — 2026-09-09 13:20 KST

Four fresh native engines: two plain, two 2-way split with the existing
112-SM compute/16-SM collective settings. Four real-content 2048-token prompts,
identical prefixes and first-token positions; independent HF reference on GPU4.

Baseline restarts agree 4/4. Split restarts agree 3/4; cross-plan blocks agree
3/4 and 4/4. The mismatch has native baseline top-two gap 0.0078125. Independent
HF agrees with both plain runs and one split at 4/4; the other split differs
at one position where **HF top-two gap is zero**. Mean HF/native KL is
2.12e-5–5.00e-5. This is weak evidence for a deterministic semantic bug, not a
quality certificate. The mismatching performance pair stays excluded.

Live captured native logits shape is **8192 × 151936**, although only positions
2047/4095/6143/8191 are selected as the four request next tokens. This confirms
the source-level prompt-wide final-head materialization in this native path.
No E2E mass is assigned to it without timing, and last-token selection would
be a trivial existing optimization rather than a new successor principle.

Artifacts: `numerical_prefill_resource_20260909_v1/` and
`hf_prefill_resource_reference_20260909_v1/`. All timing is diagnostic-only.

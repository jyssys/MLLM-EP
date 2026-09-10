# Experiment log

## 2026-09-10 — setup and safety

- Inspected active repository HEAD and created an isolated worktree/branch.
- Verified the only processes on physical GPUs 4–7 were repository-owned
  `run_utilize.sh` children. Stopped that burn before measurement; did not touch
  processes on GPUs 0–3.
- Recorded GPU UUIDs, memory and NVLink topology.
- Audited installed Qwen3 MoE layer and DeepEP HT source boundaries.

## Instrumentation smoke

- First smoke exposed that the enable flag was set too late for
  `sitecustomize`; no result was used.
- Fixed enable ordering and an empty coordinator flush path.
- Second smoke verified all four EP ranks, pairwise stream/event joins, saved
  logits, exact replays, and output token agreement. Smoke data remains ignored.

## Fresh primary runs

### Text, 107.85 seconds on four H100s

- Exact 2,363-token text prompt, 2 warmups, randomized 3 clean + 3 instrumented
  requests, one diagnostic pairwise request, one correctness request and one
  decode capture request.
- Pairwise grid: four warmups + 12 measurements for each of Dispatch, Expert,
  Combine and whole MoE.
- Policy grid: three warmups + ten measurements for every aggregation/sms point.

### Vision, 113.91 seconds on four H100s

- Exact 2,363-token input containing 2,340 vision tokens, identical protocol.

## Analysis/validation

- Collapsed exact duplicate flush payloads, then selected rank-critical duration
  once per logical observation.
- Recomputed eta from rank-critical component durations; never max-reduced
  already-computed per-rank eta.
- Built precedence-constrained perfect and contention-corrected schedule DPs.
- Corrected the aggregation oracle so an unfilled static maximum aggregation
  cap is not misclassified as a phase policy.
- `python -m pytest -q poc_modality_phase_pivots/tests`: 2 passed.
- `compileall`, analysis assertions and `git diff --check`: passed.

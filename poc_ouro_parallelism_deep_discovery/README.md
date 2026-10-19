# Ouro parallelism deep discovery

Status: `ACTIVE`. This is a resumable campaign, not a completed architecture result.
The authoritative contract requires ten substantive hours before normal completion.
The current time ledger records less than that duration.

The empirical targets are dense Base Ouro-1.4B and Base Ouro-2.6B.
The modes are fixed R4 and the pinned original-gate adaptive path.
BF16 weights remain unchanged. No new component receives training.
Only physical GPUs 4/5/6/7 belong to this campaign.
P8 remains unmeasured.

## Read first

- [Saved state](research_state.json)
- [Progress](reports/progress_log.md)
- [Next actions](reports/next_actions.md)
- [Specification snapshot](reports/spec_snapshot.md)
- [Reproduction](reports/reproduce.md)

## Current evidence

- [Prior evidence and isolation](reports/scope_and_prior_evidence.md)
- [Actual Ouro semantics](reports/ouro_semantics_audit.md)
- [Exact frontier](reports/exact_frontier_contract.md)
- [Candidate dispositions](artifacts/candidate_checkpoint.json)
- [Cost screens](reports/cost_and_headroom.md)
- [Quality boundaries](reports/quality_and_exactness.md)
- [Measured native and component CSV](analysis/checkpoint_03/)

No candidate has measured architecture E2E speedup.
Several screens reject a tested mechanism or identify direct prior overlap.
Those outcomes do not prove that every new Ouro architecture is impossible.

## Raw evidence

Each `outputs/` directory contains a unique run contract, console log, source snapshot, and guard status.
Successful model/component runs contain their own raw metrics.
Failed attempts remain versioned beside successful attempts.
The private process inventory is not a publication artifact.

The source registry retains read depth, versions, code locators, and unresolved metadata.
The generated ledger snapshots do not replace original source notes.
Paper versions and concurrent worker activity must not inflate source counts or elapsed campaign hours.

# FlashVEP Debate Note: Phase 1 Profile

> Current TP4/effective-EP4 result:
> `docs/flashvep_poc/tp4_phase1_gate_20260803.md`. The remainder preserves the
> earlier TP7 blocker report.

Date: 2026-08-03
Recommendation: **NO-GO / HOLD for Phase 2**

## Result For Debate

No Phase 1 latency sample exists. The required seven-GPU baseline failed model
parallel validation before weight loading: Qwen3-VL has 32 attention heads,
which cannot use TP=7 in vLLM 0.20.0.

Therefore all requested stage timings, rank/layer breakdowns, routed/local
expert loads, critical rank, profiling overhead, and vision/text runtime counts
are unavailable. The machine-readable CSV fields are intentionally empty,
not zero.

## Headroom Judgment

- Oracle speedup: not computable.
- Exposed non-expert fraction: not computable.
- First-tile latency: neither measured nor estimated.
- Phase 2 thresholds (>=1.15x and >=15%): not demonstrated.

The current NO-GO is due to an invalid seven-device baseline, not evidence that
FlashVEP lacks headroom. Archived TP=8 FusedMoE totals cannot answer the Phase 1
question because they neither match the device layout nor separate stages and
whole-layer wall-clock time.

## Scope Confirmation

No profiling patch, tile execution, scheduler, kernel, overlap path, placement,
or checkpoint change was made. Phase 2 was not started.

Full report: `poc_flashvep/reports/phase1_profile.md`.

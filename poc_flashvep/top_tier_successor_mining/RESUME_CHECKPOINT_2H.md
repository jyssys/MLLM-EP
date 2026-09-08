# GPU resume checkpoint — 2026-09-08 ~13:03 KST

Physical4–7 only; first resumed work11:03 KST. Burn is off during research and
never counted as measurement. Clean sequential EP screen is actively running.

## Completed

- Official MoDES1024/grid100 calibration fully completed436 points; cached
  repeat exact. Fresh128 ChartQA+128 GQA held-out quality with official thresholds.
- Official-norm SERE B1 and fixed HF B16 controls completed. Stable-vanilla
  ChartQA17/17 aggressive-SERE B1 failures recover at HF B16; GQA11/12 recover.
  This is not yet a continuous-batching or independent modality-causal result.
- Native Libra real48-layer text M128/M2048, exact32-image VL capture, first
  corrected native VL group. Two bridge lifetime/layout defects fixed with
  red/green CPU regressions and live repeats. No defective run used as science.
- DeepEP288 rank-local duplicate/sentinel/empty-source primitive tests PASS.
  Fresh renderer sanity60 unique image hashes/0% cache hits; full greedy no-op
  and zero-weight-vs-fast-sentinel parity,actual TP2/DP2/EP4 HT/Triton verified.

## First clean natural-EOS EP results (one engine per condition)

128 unique requests,3 matched cohort repetitions,6 interleaved policies.
Numbers below are request-level median paired reductions; negative means slower.

| Condition | SERE S2/.5 | SERE S4/.5 | SERE S2/.7 | MoDES70 | MoDES85 |
|---|---:|---:|---:|---:|---:|
|ChartQA offered B1/DP|−14.81%|−13.51%|−11.87%|−13.86%|−17.06%|
|GQA offered B4/DP|−12.55%|−11.80%|−11.39%|−9.40%|−11.16%|

ChartQA stock accuracy89.84%; SERE S2/.5 70.31%, S4/.5 89.84%, S2/.7 87.50%,
MoDES70 86.72%,MoDES85 84.38%. Original HF point estimates are kept separate;
the resumed EP results are not retroactively substituted into old quality runs.

## Open controls — do not declare method failure yet

1. Port overhead: token modality/TP-padding metadata is invariant within a
   forward but currently recomputed each layer. Optional cache retains EXACT
   token identity/masks, reset on every model forward; paired GPU control pending.
   Any recovery is baseline engineering, not a successor contribution.
2. Offered B is not guaranteed scheduled B: cold-image CPU preprocessing and EOS
   alter actual active requests. Separate scheduler-context diagnostics prepared.
3. Fixed-output32 control needed to separate altered generation length from
   per-step cost. No forced-output accuracy will enter natural quality Pareto.
4. Libra future-vanilla-logits replay is not100% route-exact in BF16 VL. A new
   separately labeled fixed-current-route pair gives actual prediction100% while
   keeping original predictor compute and author execution; CPU tests pass,
   native group sweep pending. No full-request E2E oracle claimed from prefill.

## Next

Finish all six EP conditions, then bounded overhead/fixed-length/scheduler
controls and remaining native VL groups+large text control. All-three screening
still precedes ranking. No Kimi or successor method is justified yet. New ACE,
PROBE,AnyExperts,MACS/XShare and the original papers remain adversarial references.

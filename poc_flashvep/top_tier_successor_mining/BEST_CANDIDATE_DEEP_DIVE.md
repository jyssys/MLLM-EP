# Final selected candidate: SERE — incremental only

All three methods received paper/code audit,live functional or quality screening,
MLLM transfer diagnostics,failure matrix,headroom assessment and prior-art attack
before final ranking. SERE is selected exactly once as the strongest **diagnosed
limitation**,not as a surviving paper successor.

Its deep dive includes official calibration,256 held-out questions,HF B1/B16,
actual EP B1/B4/B16,three natural-cohort repetitions,termination diagnostics,
same-questions S/rho controls,fixed32 output,sampled scheduler membership and
algorithmic no-op overhead. This goes beyond the exploratory quality pilot.

The largest EP B1 quality losses are19.53pp ChartQA and7.03pp GQA. Existing S4
recovers19.53pp and6.25pp,respectively,with paired E2E changes of+1.31% and+.21%
improvement relative to the aggressive original setting. This is an observed
quality-efficiency recovery,but **trivial parameter tuning** already does it.
Both settings remain slower than stock in this deployment port.

Scores assess successor evidence,not the merit of the original papers:
SERE23/60,MoDES20/60,Libra18/60. Untested Kimi generality scores0 as unestablished,
not as a failed cross-model experiment. The majority of meaningful post-screen
candidate analysis is the SERE falsification documented in the adjacent files.

No successor prototype is implemented. S4 is explicitly Baseline+TrivialFix,
not OurMethod. Kimi is not run because no material nontrivial failure survives.
This obeys the promotion condition and avoids spending GPU time merely to fill
an unconditional-looking checklist.

FINAL STATUS: **FOUND_INCREMENTAL_ONLY**.
EXACT MLLM MOTIVATION: not established; visual conditioning is present,but modality
is not proven causal after controls.
EXPECTED NONTRIVIAL QUALITY-MATCHED E2E GAIN: not established.
EXPECTED EFFICIENCY-MATCHED QUALITY RECOVERY: existing S4 already achieves the
bounded19.53pp/6.25pp recovery above; do not market it as a new principle.
WHAT COULD REOPEN IT: a repeatable material residual after original knobs,with
faithful optimized execution,matched-quality E2E headroom and Qwen+Kimi evidence.

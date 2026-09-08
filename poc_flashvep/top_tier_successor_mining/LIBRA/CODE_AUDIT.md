# Libra — code availability and faithful-port boundary

## Supplement received 2026-09-07 ~16:00 KST

The user supplied README, REPRODUCE and three supplementary SGLang diffs under
`.orca/drops/`. This supersedes the public-repository access limitation below.
The documented base is `023288645b80fb41b3eed55fd413dd69a7904593`.
The main patch includes the actual Cython planner, Qwen/GLM execution changes and
benchmark harness. The internal patch is explicitly for imbalance/route analysis;
Lina is a separate predecessor baseline, not Libra's predictor.

The newly prepared paper-only execution probe is PAUSED, UNEXECUTED and not an
evidence source. Next evidence must use or compare against these supplied paths.
The working vLLM environment remains untouched; patch application/build is in a
fresh isolated reference checkout. The original supplementary files are retained
unchanged and their SHA256 hashes recorded in SUPPLEMENT_AUDIT.md.

## Earlier access audit (historical; no longer a code blocker)

The final paper points to `https://github.com/SNU-ARC/Libra`.
On 2026-09-07, GitHub web, git remote lookup and public REST API returned not-found.
Public SNU-ARC repository inventory was downloaded; no Libra entry was found.
These are access observations, not a claim that the authors never released code.

The final PDF provides Appendix B replication and Appendix C token-sharding
pseudocode. A paper-derived implementation can be unit tested for conservation,
capacity, locality and exact expert semantics. However several scheduling details
(tie-breaking, replica lifetime and planning data structures) are not fully exposed.
Any such implementation must be labeled PAPER-FAITHFUL MECHANISM PORT, not
OFFICIAL SYSTEM REPRODUCTION. No source-based claim beyond the published
pseudocode is currently justified.

Original 235B/355B BF16 models cannot fit the allowed 4x80GB memory budget. A Qwen
30B transfer is a scaled mechanism probe, not an exact original-model speed replay.
Do not classify missing code/model-scale constraints as METHOD_FAILURE.

Next: inspect supplemental availability, implement actual next-gate predictor,
validate planner invariants and profile representative real expert replication.

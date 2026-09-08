# Completed screening artifact index

Main report:`../reports/top_tier_successor_mining.md`.
Result root:`../deepep_revalidation/results/top_tier_successor_mining_20260907_135950/`.
All relative entries below are under that result root unless stated otherwise.

| Evidence | Authoritative location | Scope |
|---|---|---|
| Final paired quality/E2E |analysis/final_screen_20260908/|Corrected MoDES and SERE;42 primary policy-condition rows|
| MoDES independent engines |analysis/modes_b16_independent_20260908/|Three engines;no cross-engine CI invented|
| HF quality |analysis/official_confirmatory_b1_20260908/|128+128 held-out;not speed|
| SERE batch control |analysis/sere_batch_control_20260908/|Fixed HF cohort versus B1|
| Existing-knob recovery |analysis/trivial_fix_attacks_20260908/|Executed paired S4/rho fixes|
| Port controls |analysis/resume_controls_20260908/|Metadata/no-op/fixed output/phase controls|
| Actual route samples |analysis/route_control_b1_20260908/ and route_control_b16_20260908/|Selected layers/early steps;not full skip fraction|
| Native VL prediction oracle |analysis/libra_eight_groups_20260908/|32inputs/8groups;not full request E2E|
| Native large text |analysis/libra_native_text48_m8192_nohooks_20260908/|No oracle hooks;80/80 first tokens|
| Native actual overlap |analysis/libra_native_nsys_20260908/|Profiled diagnostic;not clean speed|
| Official MoDES frontier |quality/modes_frontier1024_grid100/|436 unique evaluations|
| Runtime activation |analysis/final_screen_20260908/runtime_activation.csv|Four-rank real DeepEP proof|
| Hardware/time |resume_20260908/ and raw/telemetry_20260908.csv|Only4–7 in resumed turn;burn excluded|
| Figures |plots/resume_20260908/|Quality,primary Pareto,Libra oracle,independent MoDES restarts|

The commit intentionally excludes model/input tensors,images,paper PDFs,private
author supplements,cloned repositories,full request answer traces and large
Nsight binaries. They remain locally available. Curated summaries retain source
hashes and reconstruction commands; full reproduction needs those external
dependencies and authorized GPU access.

Do not reuse any folder containing INVALIDATED.json. In particular,exclude the
first UUID-dropping renderer,early VL layout/input-lifetime failures,and the
M8192 oracle-interleaved state failure. Historical CPU/pilot reports are retained
for chronology,not used as current completion status.

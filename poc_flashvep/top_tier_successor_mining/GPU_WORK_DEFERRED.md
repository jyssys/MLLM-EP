# Deferred GPU work — release checkpoint and resumption

## 2026-09-08 resume override

The user now authorizes this checklist on physical GPUs **4,5,6,7 only**.
The CPU-only restriction and earlier1/2/3/4 mapping below are historical.
Current authorization is recorded in GPU_EXECUTION_POLICY.json. Idle/final burn
is requested on4–7, is stopped before measurements, and is not research GPU time.

### Final completion state — 2026-09-08 GPU resumption

| Deferred item | Current evidence/status |
|---|---|
| Official-norm SERE quality | DONE:128 ChartQA+128 GQA,official table,three parameter settings and HF B16 control |
| SERE natural EP request timing | DONE:ChartQA/GQA×B1/B4/B16,128 questions,three randomized paired cohort reps; fixed32 and no-op cost controls also completed |
| Native Libra full48-layer text/VL | DONE within bounded bridge:32 inputs/eight groups; internal320/320 first-token agreement,HF31/32 distinct-input agreement; native no-hook M8192/source80/80 |
| Native Libra actual prediction headroom | DONE:valid eight-group frozen-current-route gain0.146%; Nsight actual overlap measured; large oracle-interleaved diagnostic INVALIDATED,not performance evidence |
| Official MoDES1024/grid100 frontier | DONE:436 unique evaluations,both targets completed,cached-point parity on resumed hardware |
| MoDES held-out quality and fast EP | DONE:corrected mask-lifetime six-condition screen plus three independent GQA-B16 engines; metadata/fixed-length/prefill-only/no-op controls DONE |
| Kimi and successor prototype | NOT_RUN:all-three ranking completed;no material nontrivial survivor. SERE selected for deeper falsification but existing S4 repairs most measured quality loss |

Final result:FOUND_INCREMENTAL_ONLY. This is not original-paper speed reproduction
or a universal rejection of these methods. Main report:
`poc_flashvep/reports/top_tier_successor_mining.md`. No GPU work auto-restarts;
the user-requested final utilize process is separate and is recorded in DELIVERY.md.

All new execution is on4–7. Native Libra captured-VL timing excludes vision
encoding and is not full request E2E. The clean EP screen uses core-token-ready
request latency,not just a serving-wave duration. Source/parity corrections are
documented in PORT_COST_CONTROL_20260908.md and LIBRA/VL_PORT_VALIDATION_20260908.md.

Original release checkpoint follows unchanged.

User instruction: CPU only from 2026-09-07 17:37 KST; confirmed afterwards.
Do not run these commands until the user separately reauthorizes GPU availability.
Only physical GPUs1/2/3/4 are eligible. No burn or automatic resource reacquisition.

| Priority / dependency | Question | Minimum experiment | Estimated GPU wall time | Decision enabled |
|---|---|---|---|---|
| 1 | Does official-norm SERE reproduce the exploratory quality loss? | Same128 ChartQA and GQA controls, original table and S/rho trivial fixes | 20–45min with four quality replicas | Separate arithmetic/port effects from algorithmic loss |
| 1 | Does SERE actually improve natural request completion? | Clean EP4 natural-EOS B1/B4/B16; fixed-output control separate; randomized paired policies | 45–90min including loads/repeats | Quality-matched real E2E / lost-speedup bound |
| 1 | Does the full supplied Libra path preserve outputs? | Exact32 VL capture, full48-layer text and VL parity to reference | 15–40min if bridge works | Accept/reject bounded port, not the paper |
| 2 after parity | Does Libra lose material overlap in real MLLM shapes? | Native vanilla/Libra matched natural groups, actual predictor vs exact-route diagnostic | 30–60min | Real prefill/TTFT headroom rather than planner counts |
| 1 | Finish official MoDES calibration frontier | Resume301 cached points with identical1024/grid100 settings | ~30–60min, cache reload included | Complete second target without changing calibration protocol |
| 2 after thresholds | Does MoDES have an unrepaired Pareto failure? | Held-out GQA/ChartQA, calibration-only trivial fix, clean fast-sentinel EP run | 45–90min | Original-method quality/speed and successor opportunity |
| Conditional, not authorized now | Does one strong causal failure generalize? | Kimi validation only after all-three screening and winner selection | Unknown until checkpoint/memory feasibility audit | Cross-model gate |

Estimates are planning ranges, not measured durations or promises. They overlap
where quality replicas can share the four devices; timing runs must be isolated.
Stop baseline setup debugging at its bounded budget rather than treating it as a
method failure. All-three milestones precede ranking, and no Kimi download/run
is needed merely to fill a checklist.

## Resume commands / provenance

`raw/modes_frontier1024_grid100_resume.log` and `interrupted_user_release.json`
identify the interrupted run. Original arguments remain in
`resume_baseline_screen.py` and `modes_frontier.py`; reuse the same output directory
to read its cache, but choose a NEW raw log filename. The handoff script has a
GPU-policy guard and was terminated. It does not restart by itself.

Do not reuse the old launcher PID. Revalidate mapping, own-process identities and
absence of conflicting jobs at the time of an explicitly authorized resume.

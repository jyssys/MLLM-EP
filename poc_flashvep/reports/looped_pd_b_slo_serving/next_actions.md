# Resume instructions

The active task is fixed-depth B SLO serving. Do not resume A/C, dynamic exit, MoE, or the paused loop-PP project.
Read the entire authoritative `../looped_pd_b_slo_serving_poc_spec.md` and the reporting skill before a new work session.
Read `research_state.json`, `state/campaign.json`, and `configs/budget.json` next.

The original clock starts at 2026-10-29 21:29:30 KST.
The hard deadline is 2026-10-30 12:29:30 KST. GPU work must stop 45 minutes earlier.
A resume never starts another 15-hour allowance.
Drop 4/8GPU extras before primary W1, matched controls, output checks, or SLO locking.

The initial qualification controller is PID3332224. The serving controller is PID3344105.
These identifiers are historical once either process exits. Check ownership and command lines before any action.
Do not start duplicate controllers or signal healthy inference jobs.
Only approved owned burners may stop. Quiet serving intentionally leaves other GPUs without burners.
Restore idle burners after clean measurements and at final closeout.

The known fresh HF discrepancies remain Ouro7/8 and Huginn6/8.
The prior frozen counts remain separate at Ouro88/100 and Huginn89/100.
These are sequence counts, not answer accuracies.
No original-model-lossless or loop-exclusive novelty GO is currently permitted.

## Current priority: diagnose B before further serving

The user requested B repair before the remaining performance study.
Repair controller PID3568550 owns the quiet diagnostic lane.
It paused CPU controller PID3344105 after its healthy inference job completed.
Supervisor PID3394636 still enforces the original deadline.
Check `state/repair_priority.json` and `/proc` before any action.
Do not resume the main controller while the diagnostic queue owns the GPUs.
Pending jobs are in `state/repair_queue/`. Completed `.done` files retain their configurations and outcomes.
The repair controller resumes the old workflow after ten idle minutes without a new heartbeat.

Qualification watcher PID3630273 manages the canonical candidate and refreshes that heartbeat.
Check its current state in `state/repair_followup.json`.
It expands through Ouro8/32/200 and Huginn33/200 requests.
The completed Ouro8-request stage passes8/8 B/control and8/8 control-repeat outputs.
The linear-only ablation still fails7/8 for B/control at that size.
Ouro's8/32/200-request stages now pass exact B/control and identical-control-repeat outputs.
Huginn's33-request stage also passes both comparisons. Its200-request stage is active.
Do not treat the canonical candidate as accepted before its complete qualification finishes.
The explicit full200-request Ouro regression commands return exit status0.
Their saved records are `configs/b_repair_ouro_full200_green.json` and `configs/b_repair_ouro_full200_repeat_green.json`.

Do not edit `repair_hooks.py` or `repair_coda.py` during this locked epoch.
`configs/repair_candidate_source_lock.json` pins those sources.
Any necessary later math-path change needs a new qualification epoch. Do not silently reuse the current passing counts.

The watcher can enter `repair_epoch.py` only after the full200-request gates.
That module also checks exact stops, fixed-depth work, source manifests, and reversed-fragment byte identity.
It keeps a separate execution epoch and uses the frozen original SLOs.
It applies the same runtime changes to DP_Mixed, PD_G1, and B.
Its rates are declared in code before serving results. Its independent seeds use crossed GPU pairs and arm order.
Its400-arrival windows use two complete cycles of all200 source prompts, not a200-row dataset labeled400.
Original failures remain closed. Do not pool original and repair-epoch timings.

Actual KV diagnostics are now available.
For the reproduced Ouro request3 case, prompt KV and token history through index124 are identical.
Original R3 historical decode KV differs in551,940 elements. The canonical candidate differs in0 elements.
Read `b_repair_diagnosis.md` and the `b_repair_*causal_KV.csv` tables.
This evidence is diagnostic. It is not a serving-speedup measurement or a Huginn result.

The actual causal-prefix test reproduced B/control mismatches: Ouro7/8 and Huginn31/33 exact sequences.
Identical-control repeats also differed: Ouro7/8 and Huginn30/33.
These results do not prove a B-specific KV corruption.
They also do not establish harmless numerical error.
One-variable probes test matrix geometry, CUDA Graph, asynchronous execution, streams, and BF16 accumulation settings.
The first launcher prototypes failed before generation. Preserve those logs and exclude them from numerical comparisons.
`repair_v2_*` and later names identify corrected launchers.
Explicit diagnostic launchers leave the frozen normal worker and private RLT source unchanged.

A proposed fix needs a new qualification epoch.
Apply it equally to PD_G1 and B. Preserve the old failure files and original SLOs.
Test the original200-request W1 workload after the small regression passes.
Reopen B performance only after exact paired output, full-depth, and KV checks pass.
Keep official-HF fidelity and novelty limitations separate.

Both original held-out B serving gates remain closed.
Ouro matches185/200 output sequences. Huginn matches193/200.
Do not re-enable B performance. Later B jobs are explicit correctness or heavy-profile diagnostics only.
The main controller continues ordinary DP/PD and PD_G1/DP_Mixed comparisons.
Huginn PD_G1 overload points at0.8 and1.6requests/s fail the existing request/transfer lifetime deadline.

The active CPU supervisor is PID3394636. The read-only process observer is PID3465875.
Verify those PIDs before any action. They can become historical identifiers after completion.
The observer never signals another process.

After the main controller exits, the supervisor imports the current extension code.
The extension preserves all200 W1 source prompts at every rate and uses whole cycles for longer windows.
It uses disjoint seeds2026106001 and2026107001 and explicitly alternates the remaining two-arm order.
It can fill ordinary DP/PD scale controls only after primary windows and with at least90 GPU minutes left.

Separate final diagnostics collect transfer events, repeat the original PD_G1 trace, and test selected prompts sequentially.
They also check32 concurrent prompt-KV/hidden transfers per model, with heavy byte fingerprints and reversed fragments.
Two-request native RLT GPU profiles run separately from clean serving.
The source of every output discrepancy remains unresolved until these actual diagnostics finish.

The final analyzer fixes EOS labels and keeps all planned arrivals in failed-run SLO denominators.
It never invents a failed run's drain duration or unobserved token timestamps.
Eight CPU contract tests currently pass. Final publication runs them again and stores JUnit output.

The main controller performs baseline-only calibration, W1 ladders, confirmations, supplements, and optional scaling.
`configs/slo_lock.json` must exist before the first held-out B capacity run.
If B/PD_G1 serving outputs differ, preserve the discrepancy and stop accepting B speedups for that model.
The model/RLT patch stays frozen at SHA-256 `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
The new private diagnostic `.pth` hook activates only with `B_EVENT_TRACE=1` and never in clean capacity jobs.

Reports and tables are incomplete until `09_final_decision.md` explicitly records closure.
The final publication must include only the new source/report paths.
Preserve the outside-index hash in `research_state.json`.

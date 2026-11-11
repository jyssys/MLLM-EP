# Primary held-out confirmation

The selection freezes before any of the new Arrival seeds runs.
The active configuration receipt is `state/primaryconfirmv2_confirmation_design.json`.
It records source configuration hashes and the freeze time.
Version 2 adds Loop28 before any held-out job starts.
The original design and its cancelled waiting-launcher receipt remain available.

The confirmation uses Arrival seeds 2026110951, 2026110952, and 2026110953.
Each start has N128, 2048/16, fixed R32, and offered load 1.971 requests/s.
The 128 prompt IDs and tokenized texts remain unchanged.
These seeds are new Arrival traces, not new texts.
The order rotates across seeds.
Every seed has a second independent Zero start.

| Label | Selection source | Reason and limitation |
| --- | --- | --- |
| zero | lifetime3_deferred2_zero | Strong full-device Native-path Zero. New N8/N32 and real child exits pass. No migration. |
| loop | quantum6_loop31_512 | Lowest observed ITL p99 among the tested full-device Loop31 tiling choices that reduce TTFT. Fixed cut, not an adaptive arm. |
| loopfour | match18_loop28_full | Exact four-Loop full-prompt continuation. It matches Token256's recurrent rows, not its complete FLOPs or Coda work. |
| token | token16_prefix1792_cap8 | Strong measured Token TTFT and goodput. Actual eight-Loop groups and full state qualify. Its relative Decode tail fails in selection. |
| tokensafe | lifetime3_spatial_sm32_token1984 | Lower-tail actual Token64 alternative. The staged Prelude actually submits 32 Loops. Its nominal eight-Loop cap is not an effective bound. |
| adaptive | early9_output16_loop | Actual primary online commit after early fragment receipt. N8/N32 state and receipt gates pass. Only one primary selection request moves. |
| lateloop | forecast8b_output16_loop | Same late-transfer forecast family as the dynamic Token comparator. Primary selection moves no requests. It is an active fallback control. |
| dyntoken | forecast8_output16_token | Actual online Token suffix control using the shared past-observation forecast and Native Decode guards. |

The `tokensafe` label identifies a selection intent.
It does not certify relative or absolute safety.
The confirmation applies the same frozen criteria to every candidate.
No single selection start establishes a GO.
The stronger full-device Zero remains the scientific denominator.
The selection-stage matched controls retain their separate causal role.

The equal-row Loop28 selection uses Seed 2026110940.
It does not use held-out results for tuning.
The disjoint-SM trials remain additional diagnostics.
If a later diagnostic is promising, it needs a separate confirmation.
Do not promote it from one favorable start.

Analyze ratios at the independent-start level.
Use the paired Arrival seed as the match.
Do not treat individual token intervals as independent repetitions.
Three seeds give limited confidence coverage.
Retain every completed start, including unfavorable results and both Zero observations.

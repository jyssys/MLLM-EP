# Stage 0R: model, gate, and cache

The model is `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
It is the Base checkpoint, not the Thinking checkpoint.
The existing checkpoint supports the original learned adaptive gate.
The selected operating point is q=0.5, min R2, max R4.
No policy changes during scheduling comparisons.

The backbone has 24 shared physical layers. Fixed R4 executes 96 logical layers.
The cache holds 96 depth-indexed K/V pairs.
Full-depth prefill preserves every prompt position's depth-specific KV.
Decode copies terminal KV rows to deeper slots before later reads.
The layout requires slot-homogeneous recurrent kernel batches.
This constraint remains in both live CDB and the simulator.

The loader checks the selected gate weight and bias against the checkpoint.
It rejects the separate, unused lookahead and preloop heads as gate sources.
Those upstream optional heads can produce missing-weight warnings. They do not determine exits in this experiment.
The loader also checks the physical GPU UUID.

The new B4 smoke produced 16 tokens per request.
It executed 180 adaptive decode loop-token operations instead of 240 fixed-R4 operations.
The first output token used full-depth prefill.
This is physical work reduction, not a measured DP scheduling speedup.

The retained v2 probe checked 360 terminal-KV copies after synchronization.
Its teacher-aligned reference met its declared numerical tolerances.
However, its short free-running reference output diverged at token index 3.
Therefore, `REFERENCE_MATCH_PARTIAL` remains a qualification.
The v3 recheck does not erase it.

The new 8192-cap recheck retained the old adaptive correct counts.
GSM8K scored 21/32. MATH-500 scored 19/32.
All 64 shared token prefixes matched their old adaptive counterparts exactly.
The quality comparison is a consistency recheck of the same CDB implementation.
It is not a comparison against an independent, higher-quality challenger.

The old Fixed-versus-Adaptive MATH result remains a quality warning.
Scheduling discovery proceeds under the v3 continuation rule.
Full GSM8K and MATH-500 quality evaluation is `NOT_RUN`.

See `dynamic_quality_reference_recovery.md` and `artifacts/quality_recheck_summary_v2.json` for denominators, uncertainty, and caps.

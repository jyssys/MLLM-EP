# Correctness contract

The pretrained weights, BF16 precision, four recurrent passes, RoPE, and causal mask remain unchanged.
Each recurrent pass has 48 independent KV slots. The full execution has 192 slots.
Pass boundaries preserve residual resets and the native normalization.

The single-GPU reference uses the checkpoint's official HF code in the existing isolated environment.
It compares full prefill with a diagonal task order on two frozen component inputs.
The masks include all committed same-depth prefix keys and only causal keys within the current chunk.
Every task checks the prefix length in all 48 stage-local cache slots before and after execution.

The component limits freeze before observation:

- Maximum absolute selected-logit difference: 0.125.
- Full hidden relative L2: 0.01.
- Per-slot K/V relative L2: 0.01.
- Selected greedy next-token IDs: identical.

Any failed component remains a failure under those limits. Do not increase a tolerance after measurement.
The HF reference is not an optimized serving baseline. SDPA and native TP4 FA3 need not be bitwise equal across engines.
Cross-engine full-logit differences and written-KV samples receive descriptive records, not an automatic bitwise PASS.

Hardware timing uses separate forced 64-token outputs. Those outputs do not certify accuracy.
Stage C requires 24 fresh complete greedy responses and paired benchmark evaluation.
Those conditional tests remain `NOT_RUN` unless the dependency, novelty, physics, and native-quality gates permit a candidate.

The earlier retained-graph natural mismatch remains an unresolved prior limitation.
This campaign uses ordinary native graphs for its quality baseline. It does not erase earlier failures.

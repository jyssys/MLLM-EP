# TEAM to LLaDA2.0-mini port audit

## Provenance

- Paper: TEAM, arXiv:2602.08404.
- Official repository: `PKU-SEC-Lab/TEAM-MoE-dLLM`.
- Pinned official commit: `e9c502e5753ce79f660371e2fb4a8666f66cae75` (2026-02-28).
- Direct LLaDA2.0 support: **not found**. The public implementation is SDAR-specific.
- Intended row label: `TEAM-PORT`, never official/native TEAM.

## Code-level audit

The official generation/model path was inspected for:

- decoded-token routed-MoE output reuse (`past_hidden_states` / `decoded_index`);
- hot token classification (`confidence >= 0.7` or distance within 3, with top-two speculative candidates unrestricted);
- four-way base/top1/top2/both speculative candidate construction;
- cold-token restricted routing through the union of necessary experts selected by unrestricted tokens.

An LLaDA2 adapter scaffold was added with unit-tested hot/cold classification, necessary-expert restriction, routed-MoE output reuse, actual post-restriction expert IDs, and official four-way candidate construction. The live adapter is deliberately labelled `TEAM-PORT-DCD-LAC`: the SDAR-specific speculative acceptance path was not executed, so it is not a faithful full TEAM result.

## Why no TEAM benchmark row was produced

The mandatory F1 n=32 gate failed first. The contract's stop rule therefore prevented further GPU rollouts, including TEAM 8→32→128. Publishing the partial adapter as TEAM quality or EP performance would conflate an unvalidated port with the official method.

`TEAM_STATUS: FAIL (adapter scaffold audited; benchmark rollout not run after upstream stop)`


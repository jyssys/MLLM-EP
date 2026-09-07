# MoE-EP assumption-mining result

This result root records the analytical/source phase of the structural
assumption search. No fresh GPU counterfactual was authorized: all scored
candidates failed the direct request-level headroom gate before execution.

Primary artifacts are versioned under
`poc_flashvep/assumption_mining/`; the report is
`poc_flashvep/reports/moe_ep_assumption_mining.md`.

The utilization burn process was kept separate from research accounting and
used only physical GPUs 1--4 via `CUDA_VISIBLE_DEVICES=1,2,3,4`.


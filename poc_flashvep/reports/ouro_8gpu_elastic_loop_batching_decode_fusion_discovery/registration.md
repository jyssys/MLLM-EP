# ELB experiment registration

The authoritative contract is `../ouro_8gpu_elastic_loop_batching_decode_fusion_poc_spec.md`.
The model is pretrained Base Ouro-1.4B at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The runtime is RLT `4a350344027960a48384626aa646f1765fda6d4f`.
The new runtime worktree is `/home/esjung/external/ouro-elb-rlt`.
The previous prefill patch and all previous verdicts remain unchanged.

## Scope

Each worker uses one physical GPU from 0–7.
There is no TP, PP, distributed model, training, early exit, or prefill C/G optimization.
Prefill uses native split loops with C512. Mixed uses native one-loop co-packing.
Decode D1/D2/D3/D4 changes scheduling boundaries, not the required four recurrent passes.

P0 audits sources, UUIDs, ownership, baseline paths, and parallel interference.
P1 tests grouped decode and lifecycle invariants before adaptive work.
P2 measures completed-token latency and paired rebatching interventions.
P3–P6 remain conditional on the preceding physical and correctness gates.

## Prospective gates

The numerical relative-L2 threshold remains 0.03.
Token agreement, hidden/logit differences, and cache agreement remain separate fields.
A cache or lifecycle bug blocks promotion regardless of timing.
P2 uses a 3–5% completed-token benefit as an exploratory screen.
An adaptive method also needs an opposing measured regime where reform helps.
Process starts and source families, not recurrent kernels, define uncertainty blocks.
At least three independent starts confirm shortlisted timing cells.
Finite drained cohorts do not establish sustainable capacity.

## Clocks and operating lanes

Clean timing uses host-delivered complete output tokens.
The decode interval starts after every member delivers its first output.
Heavy numerical captures do not supply clean latency.
CUDA service intervals, host overhead, and end-to-end wall time remain separate.
Reserved quiet and interference controls can temporarily omit burners on idle devices.
Other unused devices retain owned burners. All eight burners restart at handoff.

No adaptive policy or large quality/capacity sweep starts before the P2 crossover gate.
All failures remain in immutable attempt directories.

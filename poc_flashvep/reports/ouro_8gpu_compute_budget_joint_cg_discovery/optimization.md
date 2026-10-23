# Discrete optimization and completion accounting

The online optimizer enumerates a finite legal action set.
For one admitted action, enumeration solves the stated 0–1 choice problem directly.
A general MILP solver adds no essential capability to that one-action formulation.
This is an algorithmic equivalence, not a measured MILP speedup claim.

The candidate menu contains C64/128/256/512/768/1024/1536/2048 and legal G1/2/3/4.
The actual C can be smaller at a prompt boundary or token-budget limit.
A continuation cannot resize C.
It can change G only at its recorded recurrent frontier.
Native staging, KV reservations, and event ordering remain unchanged.

Every proposed partial action retains its missing R4 service debt.
The U2 denominator includes prelude and all predicted remaining groups.
It uses calibrated occupancy estimates, not a claim of exact expected completion latency.
Immediate completion credit comes only from an atomic R4 span or a continuation reaching R4.
A separate staged prelude executes G=0 and completes no prompt tokens.
Its planned recurrent G remains a separate trace field.
The earlier TRAIN diagnostic-label correction remains documented in `failures.md`.

U0 is a partial token-loop-work diagnostic.
It is not a valid completed-prefill speedup objective.
U1 and U2 have software definition tests.
Only U2 drives the initial measured online pilot.
U3, a genuine H2/H3 planner, and an offline MILP verifier remain NOT_RUN.
They require incremental one-step evidence before additional implementation cost is justified.

The model is not a future scheduling oracle.
It reads only prompt length, current frontier, issued state, current event readiness, and registered historical service costs.
It does not read future output IDs, future arrivals, or actual future service times.

`budget_action_trace.csv.gz` joins selected actions to measured native quanta.
Actual prelude G=0 remains distinct from planned G.
`opportunity_frequency.csv` compares model utilities on actual ready snapshots.
That table is MODEL_ESTIMATE, not measured counterfactual speedup.
`optimization_overhead.csv` records admission CPU work and joined decision counts.
Request latency includes the actual online cost of those operations.

The software suite has 17 passing tests.
The tests do not establish pretrained-model accuracy or serving capacity.

Quiet confirmation found identical actual G1/J1 plans on 144/144 requests.
The generated token IDs and output counts also matched in all 144 pairs.
The extra C search therefore changed no executed per-request plan in this confirmation.
Mean additional selector CPU work ranged from 0.07 to 0.44 ms per cohort across the four cells.
Mean additional admission CPU work ranged from 0.22 to 1.31 ms.
These CPU intervals are not additive attribution of the GPU critical path.
They do not prove that all small wall-time differences came from selector overhead.
The action-equivalence table gives the direct negative mechanism evidence.
No positive one-step incremental result justified a more complex horizon solver.

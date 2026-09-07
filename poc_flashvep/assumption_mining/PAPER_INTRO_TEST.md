# Paper-introduction tests

## Token-scoped readiness (failed promotion)

MoE expert parallelism matters because distributed inference pays both expert
compute and communication. Current runtimes optimize asynchronous dispatch but
expose one invocation-level readiness contract. That contract assumes all
tokens need the same combine boundary. In our traces, however, a giant
dispatch wait was a real rank-local event. The direct request-level removable
mass was only 1.09%, and exact route replay did not reproduce it. Existing
partial/skip and speculative systems already relax related dependencies. A
token-scoped contract is therefore an interesting semantic question, but not a
paper-sized opportunity in this evidence set.

## Engine-owned slack (failed promotion)

Continuous-batched MoE serving matters because scheduler decisions determine
how well distributed experts stay busy. Current systems put stream and
collective ownership in the worker and send a step descriptor from the engine.
This assumes the engine need not reason about communication state. Common
runtime regimes and request turnover did produce striking wave changes. Yet
request-level effects stayed at or below 2.5%, so the apparent mass was not on
the user critical path. Existing serving schedulers already address admission
and locality but not this exact ownership boundary. A future trace may reveal
a new problem, but this branch has no direct headroom for a method.

# Experiment log

Result root: `poc_rankfanout/results/rank_fanout_ep_communication_poc_20260910_192050/`

| Phase | Fresh GPU evidence | Controls | Outcome |
|---|---:|---|---|
| Runtime path | one full-model profiled execution/backend, followed by repeated runs | source hash, live call hook, placement dump | AGRS and DeepEP HT paths verified |
| Synthetic smoke | M=256, F1/F4, both backends | same expert/rank histogram | expanded to full matrix |
| Synthetic matrix | 5,040 measured rows | 7 M values × F1–F4 × 2 backends × 30 reps × 3 process restarts | robust crossover at M=4096/8192 |
| Real route capture | 8,640 source-rank/layer rows | 3 restarts, 5 workloads, all 48 layers, exact router top-k hook | fanout concentrated near F3/F4 |
| Clean static serving | 300 measured request rows | 5 restarts, identical requests, 3 measured iterations, two DP requests | outputs exact; substantial restart-state drift |
| Exact real-route replay | 24,000 measured rows | bit-identical route/input per backend, 240 shapes, 10 reps, 5 process restarts | stabilized dynamic headroom small |
| Selector test | train restart 2, held-out restarts 3–5 | one-dimensional threshold fixed on train | −0.040% vs best static |

DeepEP LL was not retried: it is optional in the spec and the same installed
stack already failed two bounded attempts at 8,192 and 1,024 max-batched-token
budgets with the `nvshmem_qp_depth` assertion. This exclusion does not weaken
the required AGRS-versus-DeepEP-HT decision.

The first replay restart had a cold AGRS dispatch path (152.379 ms aggregate
dispatch versus 32.465–35.456 ms in stabilized restarts 3/5). It is retained
in raw data but excluded from the stabilized headline. Restart 4 is retained
as an adverse runtime-state case and determines the reported optimistic max.

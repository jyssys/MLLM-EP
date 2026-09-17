# H1/H2 Follow-up Summary

The deep dive changes the interpretation of the earlier H3 result: one-owner placement had only ~0.3% headroom, but the actual divisible straggler upper bound is 6.11% EP4 and 12.37% EP8 routed-MoE stage time. Small-budget exact replication captures only 3.87% EP4 under the stronger global-static policy; early dLLM persistence reaches 3.46% EP4 and 5.86% EP8 and is not better than global popularity. H2 becomes more rank-volatile at EP8 but remains physically diluted (0.21% P95 full-stage effect).

```text
H1_STRAGGLER_SEVERITY:
PASS

H1_CRITICAL_RANK_PERSISTENCE_EP4:
PASS

H1_CRITICAL_RANK_PERSISTENCE_EP8:
PASS

H1_TOP_EXPERT_EXCESS_CONCENTRATION:
PASS

H1_PERFECT_BALANCE_HEADROOM_EP4:
6.11% aggregate routed-MoE stage (5.22% P50)

H1_PERFECT_BALANCE_HEADROOM_EP8:
12.37% aggregate routed-MoE stage (11.17% P50)

H1_GLOBAL_STATIC_REPLICATION_HEADROOM_EP4:
3.87%

H1_EARLY_PERSISTENCE_REPLICATION_HEADROOM_EP4:
3.46%

H1_EARLY_PERSISTENCE_REPLICATION_HEADROOM_EP8:
5.86%

H1_TRUE_EP2_REPLICA_DIRECTION:
NOT_RUN

H1_VERDICT:
HOLD

H2_EP4_DESTINATION_JACCARD_P50:
0.750

H2_EP8_DESTINATION_JACCARD_P50:
0.600

H2_EP8_FULL_WORKLOAD_P95_EFFECT:
0.21% routed-MoE stage

H2_VERDICT:
HOLD

MAIN_INTERPRETATION:
Persistent EP stragglers have real upper-bound headroom, especially at EP8, but the implementable exact-replication headroom is modest and mostly explained by globally hot experts rather than additional same-block refinement information. H2 rank transitions increase at EP8 without a material full-workload cost.

DO_NOT_IMPLEMENT_PRODUCTION_METHOD_AUTOMATICALLY:
true
```

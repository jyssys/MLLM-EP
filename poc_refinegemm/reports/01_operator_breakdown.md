# Routed-expert execution pie

The execution-pie gate passes strongly. Collapsing each traced stage with the
critical owner rank and mapping it to clean request wall gives the following
observer-attributed upper bounds:

| task | router | dispatch | routed expert | combine | shared | whole MoE |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 6.43% | 12.07% | **29.47%** | 6.41% | 2.98% | 59.52% |
| HumanEval | 7.22% | 13.32% | **29.55%** | 6.66% | 3.50% | 62.81% |

These percentages are not additive removable work: tracing and clean request
timing are deliberately separated. They show that a perfect routed-expert
replacement could matter, and justify the kernel study. They do not imply that
29.5% is removable or that rank rows may be summed as latency.

Machine evidence: [OPERATOR_BREAKDOWN.csv](../OPERATOR_BREAKDOWN.csv). Figure:
[01_routed_expert_e2e_pie.png](../figures/01_routed_expert_e2e_pie.png).


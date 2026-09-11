# Quality-latency Pareto

## Quality axis

The primary quality screen uses full-model 16-token greedy answers and the
native GQA/ChartQA answer normalization, not four-token agreement.  The table
reports 32 held-out requests.

| Policy | Visual assignment drop | Exact short output | Task score |
|---|---:|---:|---:|
| Stock | 0.00% | 100.0% | 24/32 |
| Full semantic 20% | 18.61% | 100.0% | 24/32 |
| Full semantic 30% | 27.15% | 100.0% | 24/32 |
| Full semantic 40% | 35.81% | 96.875% | 24/32 |
| Full semantic 50% | 45.04% | 93.75% | 24/32 |
| Fixed visual K=6 | 22.52% | 100.0% | 24/32 |
| Fixed visual K=4 | 45.03% | 93.75% | 25/32 |
| Actual-contribution 30% | 27.02% | 96.875% | 25/32 |
| EP-refined semantic 20% | 18.61% | 100.0% | 24/32 |
| EP-refined semantic 30% | 27.15% | 96.875% | 25/32 |

The +1 scores of fixed K=4 and contribution selection are single-request
flips and are not interpreted as accuracy improvements.  The same data show
that 40–50% is beyond the strict output-preserving frontier.  A substantially
larger benchmark would be required to label that regime quality-safe.

## Latency axis

The latency axis is the median of 30 randomized repetitions on each of three
real image routes at Qwen layer 24.  The TTFT column is an optimistic Amdahl
projection using a measured 62.191% MoE share from the prior 16K clean run.

| Policy | Visual drop | MoE reduction | Projected TTFT |
|---|---:|---:|---:|
| Fixed K=6 | 25.00% in replay | 9.79% | 6.09% |
| Global semantic 20% | 19.90% | 12.21% | 7.60% |
| Global semantic + EP 20% | 19.90% | 15.94% | 9.92% |
| Global semantic 30% | 29.46% | 16.96% | 10.55% |
| Global semantic + EP 30% | 29.46% | **21.13%** | **13.14%** |

The replay's 29.46% drop is slightly above the held-out quality run's realized
27.15% because image/token composition differs.  It is not legitimate to
pair the maximum 45% quality stress point with an unmeasured 45% replay.

## Frontier interpretation

The combined candidate crosses the specification's 12% projected threshold,
but that number does not represent a 13.14% new EP contribution.  The semantic
schedule already supplies 10.55 points; assignment-preserving physical-rank
refinement supplies 2.59.  Early and late layer controls reduce that increment
to 0.53 and 0.57 points.

Moreover, the strict quality and speed gates do not intersect.  The 20%
refined policy keeps 32/32 outputs but projects 9.92% total TTFT; the 30%
policy projects 13.14% but changes one output relative to stock.  The changed
answer happens to repair one baseline error in this small cohort, which is not
credible evidence that the policy is quality-improving.

The semantic-only frontier is itself attacked by fixed K=6, which recovers
82.9% of strict quality-safe work removal.  After selector, mask, rank-load
aggregation, and production ragged-K overhead, the residual EP-specific
frontier can only shrink.  This is why the PoC is a NO-GO despite a combined
operator point above 12%.

## Evidence boundary

- Quality policies do not save runtime; all experts still execute.
- Replays save actual DeepEP/expert work but are captured-layer operator runs.
- TTFT is projected, not observed.
- No result claims statistically stable benchmark equivalence from only 32
  requests.

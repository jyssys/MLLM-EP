# 04 — MoE/EP physical-cost specificity

The observed 0.9/0.825/0.825 schedule is a decisive negative control for
novelty: it reads confidence/mask progress and **no** routed expert, rank,
dispatch, expert execution, or combine feature. Removing all EP inputs from
this policy changes nothing. Its measured gains are therefore recoverable by
a generic training-free dLLM acceptance rule. A separate D/E EP-cost-aware
controller was not built, so its possible incremental headroom remains
**unmeasured**, not proven zero. The required >=3--5% quality-matched extra
direct E2E from EP features has not been demonstrated.

The current dense-block runtime still executes 32 physical rows per ready
request while it is refining. Every routed invocation has `8*M` exact
token-expert branches; accepting more positions mainly removes entire future
NFE/ready requests rather than reducing within-forward DeepEP rows.
Layer-16 observer geometry verified a real 992-row routed wave with 7,936
branches, 5,857 remote, rank-load vector `[2086,1557,2827,1466]`, mean
destination-rank fanout 3.17, and rank CV 0.273. Across baseline trace phases,
median remote fraction stayed roughly 0.745/0.756/0.758 and mean rank fanout
roughly 2.97/3.02/3.03. Ready-pool M fell 1,024→832→160, while tiny
expert fraction rose 0.278→0.284→0.536. This confirms *physical EP work*
varies, but does not show it predicts a quality-safe budget beyond confidence
and physical M. Source:
[observer trace summary](../results/observer_base_gsm32/TRACE_SUMMARY.json).

The [EP specificity ablations table](../EP_SPECIFICITY_ABLATIONS.csv) is
header-only by design: there is no EP-conditioned quality–latency ablation to
report. Observer timings were inflated (+57% baseline, +112% phase) and are
unsuitable for estimating a 3% incremental serving gain. Earlier same-substrate
[routed expert E2E attribution](../../poc_refinegemm/reports/01_operator_breakdown.md)
was about 29.5% as an observer-attributed *upper-bound mass*, not a removable
EP-only budget for this policy. Rank balancing's earlier small economic pie is
not silently added back here.

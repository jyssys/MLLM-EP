# 08 — Quality–latency Pareto

The measured-only [QUALITY_LATENCY_PARETO.csv](../QUALITY_LATENCY_PARETO.csv)
records task, matched dataset size, generation setting, mini, benchmark score,
NFE, clean pool BCT, and gain versus the matched baseline median. The
[GSM8K full](../figures/quality_latency_nfe_gsm8k1319_mini32.png),
[GSM8K 512](../figures/quality_latency_nfe_gsm8k512_mini32.png), and
[HumanEval full](../figures/quality_latency_nfe_humaneval164_mini16.png)
plots show measured score versus BCT/gain/NFE. These are descriptive
scatterplots, not bootstrap quality certificates or perfect oracles.

| Promoted point | Quality delta vs matched baseline | Paired quality lower 95% bound | Clean BCT gain | NFE reduction | Promotion gate |
|---|---:|---:|---:|---:|---|
| GSM8K full phase 0.9/0.825/0.825 | +0.152 pp | -0.455 pp | 6.89%, one restart | 8.09% | ≤0.5 pp supported, but <10% E2E |
| HumanEval full same phase | +0.610 pp | 0 pp | 3.95%, one restart | 8.78% | <8--10% E2E |
| GSM8K full static 0.825 | -0.303 pp | -0.986 pp | 8.33% descriptive median | 11.25% | only ≤1 pp supported; static frontier competes |
| GSM8K full static 0.800 | -0.152 pp | -0.834 pp | -8.54% descriptive median | 14.51% | work shrinks but BCT does not |

The n=512 phase median showed a nominal 12.93% BCT reduction, but its
paired quality lower bound was -1.367 pp; it is not a quality-safe 0.5/1.0-pp
promotion point. Neither promoted task meets `≤0.3 pp + ≥8%`, `≤0.5 pp +
≥10%`, or `≤1 pp + ≥15%` direct E2E thresholds. No useful point required
>2 pp loss; rather, the systems/EP-specific gain was too small or uncertain.

NFE decreases cannot be used as a replacement for MoE time or routed
assignment measurement. Static 0.800 reduced NFE 14.5% on the full GSM8K
set but had no repeated-median BCT benefit. Clean per-action dispatch,
expert, combine or assignment counts were not available without very high
observer overhead, so requested quality-versus-MoE/EP-work panels were **not**
drawn with fabricated `8*M*NFE` values. A future low-overhead trace is needed
to populate those axes safely.

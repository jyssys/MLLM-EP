# Trivial-fix attacks

| Attack | Fresh result | Fraction of useful oracle recovered | Judgement |
|---|---|---:|---|
| Keep natural/global interleaved order | Warmed max-token-1 median BCT 1.129 s, best clean policy | All useful request-level benefit | Strongest trivial baseline |
| High/low resolution / vision bucket | 1.211 s at max-token-1; 2.529 s at fixed-16 | No stable advantage | Reject |
| Short/long LM bucket | 1.651 s at max-token-1 (46.3% slower than global); 2.793 s at fixed-16 | Negative at large cohort | Reject |
| Text/image split | 1.238 s at max-token-1; 2.498 s at fixed-16; 3.977 s at natural max-32 | At most 5.4% diagnostic BCT vs global | Below 8% gate; multi-token output mismatch |
| Single/multi-image split | Lowest attention and MoE CUDA totals in 3/3 observer restarts | Already supplies the common LM-stage winner | Leaves only 0–4.7% vision increment |
| Random order | 1.235 s warmed max-token-1 median | No stable superiority | Reject |
| Full-shape warmup | Removes first-large-cohort confound but policy CV remains 7.3–18.1% | Prevents false positives, not a method | Required control |
| Exact actual-token DP balancing | Reduces batch-128 DP max/mean from as high as 1.275 to 1.00002 | Removes a large false grouping effect | Known/simple load-balancing control |

The strongest alleged hierarchy benefit is already bounded below 8%. Simple
global interleaving or a single/multi bucket covers the useful module behavior;
there is no residual benefit for a complex adaptive yield graph.

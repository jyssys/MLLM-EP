# Track C — MoE-aware CP degree

## Support and design

This is a four-GPU fixed-fleet prefill study on cached `deepseek-ai/DeepSeek-V2-Lite-Chat`, BF16, native PCP, EP off:

- PCP1: four independent one-GPU replicas.
- PCP2: two independent two-GPU replicas.
- PCP4: one four-GPU replica.

Independent replicas are required because current CUDA PCP rejects managed DP greater than one. Every process retained `CUDA_VISIBLE_DEVICES=4,5,6,7` and was assigned a disjoint logical subset. Exact prompt-token counts, an arrival barrier, two warmups, three measurements, and one-token greedy output were used.

## Clean request results

Numbers are median fleet wall in milliseconds; the winner is bold.

| Length | Concurrency | PCP1 | PCP2 | PCP4 | Winner |
|---:|---:|---:|---:|---:|---:|
| 8K | 1 | 119.60 | 87.10 | **86.71** | 4 |
| 8K | 4 | **121.87** | 168.24 | 235.38 | 1 |
| 8K | 16 | **457.34** | 572.85 | 750.75 | 1 |
| 16K | 1 | 256.39 | 168.16 | **117.15** | 4 |
| 16K | 4 | **259.78** | 315.84 | 421.82 | 1 |
| 16K | 16 | **969.61** | 1225.81 | 1493.10 | 1 |
| 32K | 1 | 600.11 | 377.97 | **249.30** | 4 |
| 32K | 4 | **605.68** | 714.17 | 914.72 | 1 |

Greedy first-token IDs agreed across PCP degrees for every regime.

The best static PCP1 sum is 3390.38 ms. The per-regime lower envelope is 2867.44 ms, a 15.42% raw oracle gain. A length-only selector reaches 3276.72 ms, leaving 12.49% to the oracle.

## Causal/trivial-fix attack

The obvious fixed-fleet rule `PCP4 when concurrency=1; PCP1 otherwise` exactly matches every per-regime winner and recovers 100% of oracle headroom. This is replication versus single-request CP scaling, not a MoE-aware effect.

To test whether content-dependent MoE routing changes the choice, the same exact 8K/16K shapes at concurrency two were rerun for natural, code, math, and repetitive token streams. PCP2 won all eight comparisons. Its margin over second place was 35.33--38.94% at 8K and 28.65--29.56% at 16K. Content did not change the winner.

## Capacity and limitations

PCP4 materially lowers isolated-request TTFT, while PCP1 preserves four-way fleet concurrency. This is a normal capacity/latency trade-off. The screen did not capture router histograms, and the primary Qwen3-VL MLLM model is unsupported by current PCP, so no modality or route-specific claim is made. A single restart is sufficient for this kill because the trivial rule exactly fits all eight base regimes and the content control has very large, consistent margins.

## Gate

`NO_GO`: the 15.42% raw oracle is fully recovered by a simple concurrency rule, and matched content does not change the optimal degree. The successor would be a deployment knob, not a paper-level MoE-aware scheduling problem.

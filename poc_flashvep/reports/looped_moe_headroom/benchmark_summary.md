# Baseline benchmark summary

Evidence: LIVE_MEASURED native model-only timing on physical GPU0.
Every forward cell has three starts, five warmups, and twenty repeats per start.
All parameters remain BF16.
The runtimes use no generation KV cache.

| Model | Input tokens | Clean forward ms [95% CI] | Expert span ms | Observer / clean | Expert / observer | Expert / clean, optimistic | Peak allocated GiB | Validity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Foil-1 | 128 | 31.55 [30.26, 33.06] | 4.53 | 1.171 | 12.3% | 14.4% | 1.12 | PARTIAL_OBSERVER_OVERHEAD |
| Foil-1 | 512 | 31.03 [30.31, 32.30] | 5.03 | 1.106 | 14.6% | 16.2% | 1.20 | PARTIAL_OBSERVER_OVERHEAD |
| Foil-1 | 2048 | 32.02 [31.13, 33.09] | 5.93 | 1.072 | 17.3% | 18.5% | 1.49 | QUALIFIED_EVENT_SPAN |
| IFM MoE Huginn | 128 | 207.80 [206.35, 209.64] | 48.82 | 1.097 | 21.4% | 23.5% | 16.75 | QUALIFIED_EVENT_SPAN |
| IFM MoE Huginn | 512 | 215.89 [214.04, 218.68] | 56.54 | 1.067 | 24.5% | 26.2% | 17.69 | QUALIFIED_EVENT_SPAN |
| IFM MoE Huginn | 2048 | 229.46 [227.94, 231.54] | 69.44 | 1.069 | 28.3% | 30.3% | 21.40 | QUALIFIED_EVENT_SPAN |
| IFM MoE Ouro | 128 | 217.92 [212.20, 228.56] | 48.94 | 1.055 | 21.3% | 22.5% | 16.74 | QUALIFIED_EVENT_SPAN |
| IFM MoE Ouro | 512 | 219.80 [219.72, 219.93] | 57.68 | 1.065 | 24.6% | 26.2% | 17.68 | QUALIFIED_EVENT_SPAN |
| IFM MoE Ouro | 2048 | 234.04 [231.84, 235.87] | 71.94 | 1.114 | 27.6% | 30.7% | 21.40 | PARTIAL_OBSERVER_OVERHEAD |


| Model | Prefix | Output tokens | E2E ms [95% CI] | Mean TTFT ms | Mean TPOT ms | Mean output tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| Foil-1 | 128 | 16 | 1000.4 [980.8, 1028.2] | 31.2 | 64.6 | 16.00 |
| Foil-1 | 128 | 64 | 3970.8 [3920.5, 4058.9] | 31.2 | 62.5 | 16.12 |
| Foil-1 | 512 | 16 | 998.5 [985.8, 1016.6] | 31.6 | 64.5 | 16.03 |
| Foil-1 | 512 | 64 | 4035.6 [3939.8, 4215.0] | 31.4 | 63.6 | 15.87 |
| IFM MoE Huginn | 128 | 16 | 3364.9 [3339.7, 3402.4] | 213.4 | 210.1 | 4.76 |
| IFM MoE Huginn | 128 | 64 | 13484.2 [13367.2, 13595.4] | 206.1 | 210.8 | 4.75 |
| IFM MoE Huginn | 512 | 16 | 3474.5 [3427.4, 3511.6] | 225.0 | 216.6 | 4.61 |
| IFM MoE Huginn | 512 | 64 | 13877.1 [13772.1, 14036.8] | 214.7 | 216.9 | 4.61 |
| IFM MoE Ouro | 128 | 16 | 3577.5 [3427.6, 3875.0] | 222.9 | 223.6 | 4.49 |
| IFM MoE Ouro | 128 | 64 | 14273.5 [13684.4, 15430.4] | 216.5 | 223.1 | 4.50 |
| IFM MoE Ouro | 512 | 16 | 3664.1 [3515.5, 3936.9] | 225.9 | 229.2 | 4.38 |
| IFM MoE Ouro | 512 | 64 | 14645.6 [14134.9, 15620.0] | 228.8 | 228.8 | 4.38 |


No candidate E2E speedup was measured.
Oracle and causal quality-probe wall times are not comparable clean benchmarks.
The model families have different runtimes, so cross-model raw latency is not a controlled architecture comparison.

# EP characterization and memory

## Recurrent versus one-off execution

The following table uses B1 and prefix 128.
Rank imbalance comes from exact executed top-2 routing counts.
API spans come from one cumulative-critical rank in a separate observer.
They are not clean E2E critical-path percentages.

| Model | EP | Core? | Calls | Mean rank CV | Mean max/mean | Expert span ms | Complete EP span ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | True | 112 | 0.794 | 2.199 | 46.37 | 221.73 |
| IFM Ouro | 8 | True | 112 | 1.232 | 3.800 | 43.09 | 229.24 |
| IFM Huginn | 4 | False | 16 | 0.723 | 2.086 | 6.37 | 30.07 |
| IFM Huginn | 4 | True | 96 | 0.798 | 2.262 | 41.42 | 184.90 |
| IFM Huginn | 8 | False | 16 | 1.183 | 3.611 | 6.48 | 31.23 |
| IFM Huginn | 8 | True | 96 | 1.152 | 3.580 | 39.45 | 191.44 |
| Foil-1 | 4 | False | 2 | 0.281 | 1.422 | 0.33 | 2.87 |
| Foil-1 | 4 | True | 16 | 0.242 | 1.357 | 3.07 | 25.29 |
| Foil-1 | 8 | False | 2 | 0.316 | 1.516 | 0.32 | 2.69 |
| Foil-1 | 8 | True | 16 | 0.330 | 1.596 | 2.99 | 23.57 |

Unequal rank row counts do not directly equal unequal GPU execution times.
Small expert batches retain launch and grouping costs.
The replica model's linear row-time conversion is therefore optimistic.
Huginn and Foil one-off calls remain in every whole-model cost.
The study cannot remove those calls through recurrent-core placement.

## Real replica memory and initialization

| Model | EP | Extra copies/site | Total GiB | Max rank GiB | Init max-rank s |
| --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 1 | 0.584 | 0.584 | 0.962 |
| IFM Ouro | 4 | 4 | 2.338 | 0.584 | 3.630 |
| IFM Ouro | 8 | 1 | 0.584 | 0.584 | 0.936 |
| IFM Ouro | 8 | 8 | 4.676 | 0.584 | 6.424 |
| IFM Huginn | 4 | 1 | 0.250 | 0.250 | 0.279 |
| IFM Huginn | 4 | 4 | 1.002 | 0.250 | 1.550 |
| IFM Huginn | 8 | 1 | 0.250 | 0.250 | 0.393 |
| IFM Huginn | 8 | 8 | 2.004 | 0.250 | 3.265 |
| Foil-1 | 4 | 1 | 0.009 | 0.009 | 0.017 |
| Foil-1 | 4 | 4 | 0.035 | 0.009 | 0.062 |
| Foil-1 | 8 | 1 | 0.009 | 0.009 | 0.018 |
| Foil-1 | 8 | 8 | 0.070 | 0.009 | 0.149 |

The control allocates original BF16 expert tensors and verifies matching hashes across ranks.
It does not execute replica-aware dispatch.
The copy layout is fixed for memory validation.
Candidate placement bytes are calculated separately from exact per-expert tensor sizes.
No candidate assigns work to an absent modeled expert copy.

## Compute/communication operator control

| Model | EP | Start | Sequential ms | Two streams ms | Operator ratio | Output check |
| --- | --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 4_baseline_s0_v1 | 0.4247 | 0.5059 | 0.840× | NOT_RECORDED_OLDER_CONTROL |
| IFM Ouro | 4 | 4_shape_quality_v2 | 0.5630 | 0.6224 | 0.905× | True |
| IFM Ouro | 8 | 8_baseline_s0_v1 | 0.4260 | 0.4856 | 0.877× | NOT_RECORDED_OLDER_CONTROL |
| IFM Ouro | 8 | 8_shape_quality_v2 | 0.4219 | 0.4973 | 0.848× | True |
| IFM Huginn | 4 | 4_baseline_s0_v1 | 0.4208 | 0.4684 | 0.898× | NOT_RECORDED_OLDER_CONTROL |
| IFM Huginn | 4 | 4_shape_quality_v2 | 0.5178 | 0.5959 | 0.869× | True |
| IFM Huginn | 8 | 8_baseline_s0_v1 | 0.4438 | 0.4977 | 0.892× | NOT_RECORDED_OLDER_CONTROL |
| IFM Huginn | 8 | 8_shape_quality_v2 | 0.4221 | 0.5245 | 0.805× | True |
| Foil-1 | 4 | 4_baseline_s0_v1 | 0.2111 | 0.2474 | 0.853× | NOT_RECORDED_OLDER_CONTROL |
| Foil-1 | 4 | 4_shape_quality_v2 | 0.2771 | 0.2448 | 1.132× | True |
| Foil-1 | 8 | 8_baseline_s0_v1 | 0.1872 | 0.2240 | 0.836× | NOT_RECORDED_OLDER_CONTROL |
| Foil-1 | 8 | 8_shape_quality_v2 | 0.2108 | 0.2877 | 0.733× | True |

The ratio uses the maximum rank-local mean in each complete operator condition.
Each condition has ten warmups and twenty measurements.
The two source runs use separate process starts.
The original control did not record an operator output comparison.
The later quality control adds that check and passes it.
The control tests independent native tensors, not two live asynchronous requests.
Request-level critical-path saving remains NOT_MEASURED.
Most tested controls regress when the streams compete for GPU resources.
Isolated positive rank results do not justify an E2E claim.

## Route lookahead

The prior-pass diagnostic compares authoritative top-2 routes at sampled positions.
It never changes routing or dispatches future activations.
`summary/route_false_positive_cost.csv` records false-positive edges and unused predicted expert sets.
Potential copy bytes describe a structural diagnostic, not actual allocated prefetch memory.
The gate needs the next pass's actual hidden state.
Observed route similarity cannot remove that dependency.

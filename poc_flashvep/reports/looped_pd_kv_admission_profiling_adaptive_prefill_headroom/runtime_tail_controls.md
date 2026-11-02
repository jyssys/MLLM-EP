# Runtime tail controls

Status: MEASURED_EXPLORATORY_CONTROLS. Independent-start validation remains separate.

## Conditions

Each row uses 24 measured requests.
Ouro uses 6144 input tokens and 64 output tokens at target QPS 2.0.
Huginn uses 2560 input tokens and 32 output tokens at target QPS 1.0.
Generation is greedy and ignores EOS.
Model weights, BF16 precision, fixed depth, numerical patches, prompts, seeds, and arrival offsets remain unchanged within each model.

The asynchronous writer preserves the evidence records.
It batches durable writes outside the scheduling thread.
Warm-up uses real native requests before the measured interval.
It does not remove recurrent work from measured requests.

## Measured controls

| Model | Evidence writer | Additional graph warm-up | Request TPOT p95, ms | Token ITL p99, ms | New measured-phase graph captures |
| --- | --- | --- | ---: | ---: | ---: |
| Ouro | Synchronous | No | 41.026 | 75.975 | 2 |
| Ouro | Asynchronous | No | 57.614 | 228.331 | 5 |
| Ouro | Asynchronous | Yes | 41.586 | 45.443 | 0 |
| Huginn | Synchronous | No | 134.740 | 279.289 | 5 |
| Huginn | Asynchronous | No | 159.672 | 316.088 | 5 |
| Huginn | Asynchronous | Yes | 126.488 | 131.537 | 0 |

Sources are the six `raw/graph_control_*_n24/requests.jsonl` files.
Their `graph_captures_*.jsonl` files record actual native captures.
The graph controls retain their immutable configuration and source archives.

These single-start values have no independent-start confidence interval.
The asynchronous writer alone does not consistently improve these controls.
Batch composition and lazy capture can change when scheduling overhead changes.
Do not describe these rows as a universal writer speedup.

## Remaining Huginn variation

The warmed Huginn control and the warmed primary QPS 1.0 pilot differ only in run name and phase label.
Both use zero new measured-phase graph captures.
Their measured tails still differ.

| Run | Token ITL p99, ms | Five-row recurrent submissions, including warm-up |
| --- | ---: | ---: |
| `graph_control_huginn_warm_async_q1p0_n24` | 131.537 | 0 |
| `warm_epoch_huginn_c2560_q1p0_pilot24` | 240.893 | 189 |

Both tails cover the whole measured run.
The central arrival interval remains a separate field in the causal gate table.
Do not treat their difference as an isolated causal effect.

The native `RecurrentGraphs.select` method rejects query counts above its configured row limit.
The current Huginn graph limit is four rows.
Five-row execution therefore uses the eager fallback path.
This is a source-verified execution distinction, not a measured estimate of its exclusive latency cost.

The new D-slot control limits active Decode to four rows.
It preserves the graph implementation, precision, weights, depth, native credits, and ACK rules.
Its separate output, byte-identity, and full-depth gates precede performance expansion.

Alternative causes remain possible.
KV admission can remain binding after the tail improves.
CPU submission and scheduling variation can also retain tail stalls.
The final report must use actual control results and independent starts.

## Interpretation

Generic runtime overhead can obscure Decode margin.
Removing lazy capture does not prove P compute saturation.
A restored Decode SLO does not prove usable borrowing capacity.
The target still requires the same-start P, D, KV, and actual-work evidence.

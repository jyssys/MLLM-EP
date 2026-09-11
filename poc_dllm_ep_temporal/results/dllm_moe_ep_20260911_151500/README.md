# Result bundle

This directory contains the evidence for the dLLM MoE EP temporal PoC.

- `runtime_audit/`: per-rank stock/true-EP/single-GPU audits, profiler traces,
  hidden boundaries, and logits.
- `runs/`: three clean and three traced independent restarts.
- `peer_copy.csv`: all 12 directed physical-GPU pair measurements for one
  12-MiB expert.
- `oracle_summary.json`: authoritative scalar results, including absolute
  implementation-independent upper bounds.
- `replication_oracle.csv`: H=1/2/4/8 costed replica results.
- `batching_iterations.csv`: per-denoising-iteration batching costs.
- `analysis/`: derived CSV/JSON summaries and 16 required plots.

Timing fields in route traces are observer-heavy. Clean request timing is the
economic denominator; structural trace timing is not presented as production
latency. The future-aware batching policy is a bounded offline search, while
the separately reported fractional lower bound is the true optimistic absolute
bound.

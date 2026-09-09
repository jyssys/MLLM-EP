# Correctness validation

| Workload | Requests | Exact full greedy sequence agreement | First-token agreement | Performance use |
|---|---:|---:|---:|---|
| Warmed max-token-1 | 128 | 128/128 | 128/128 | Primary clean BCT evidence |
| Fixed-16, ignore EOS | 128 | 105/128 | 126/128 | Diagnostic only |
| Natural EOS, max-32 | 128 | 107/128 | Not used as a correctness claim | Diagnostic only |

The request set, images, prompts and deterministic sampling parameters are
identical across policies. The later-token differences are consistent with
small order/batch-shape-dependent numerical changes being amplified by
autoregressive decoding; this PoC did not capture fixed-prefix logits to prove
that mechanism. It therefore does not call the divergent runs semantically
equivalent and excludes their latency from any positive speedup conclusion.

This conservative exclusion strengthens the no-go result: even before the
correctness filter, the best diagnostic fixed-16 and natural-EOS improvements
over P0 are only 2.8% and 5.4%, respectively.

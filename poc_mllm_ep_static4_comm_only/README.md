# Static-4 communication-only dispatch slicing PoC

This bounded PoC tests one fixed policy: split each prefill MoE layer's routed
DeepEP dispatch into four byte-balanced slices, run the resident decode's full
MoE path after slice 2, concatenate the received prefill rows, and execute the
prefill expert kernel exactly once. It does not implement adaptive slicing,
modality logic, an SLO controller, or a production scheduler.

The measured result is `STATIC4_OVERHEAD_DOMINATES`: communication granularity,
correct reconstruction, and one-shot expert execution were demonstrated, but
the stock-API/Python composition added 199.64% mean prefill wall time and made
the victim step slower. See [final_summary.md](reports/final_summary.md) and
[reproduction.md](reports/reproduction.md).

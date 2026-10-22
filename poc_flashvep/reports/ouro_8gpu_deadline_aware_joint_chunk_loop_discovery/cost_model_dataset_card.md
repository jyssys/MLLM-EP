# Cost-model dataset card

The input-only source manifest is `workload_manifest.json`.
It freezes 726 prompt variants from public Alpaca and arXiv sources.
Context variants of one article retain one family and one partition.
The conversation proxy is not ShareGPT.
Long contexts use source text, not repeated text padding.

Partitions are TRAIN, VALIDATION, CALIBRATION, HELDOUT, and FINAL_ONLINE.
The extractor asserts source-family disjointness.
Independent run IDs also remain within one partition.
FINAL_ONLINE is not consumed by the completed cost-model or quality work.

The main shape menu is P512/N1, P2033/N4, and P8177/N8.
The registered full source manifest contains additional context variants that did not execute in this sparse matrix.
Unused variants are not measured coverage.
Each cost cohort contains one long prompt and short background prompts.
The outputs are natural greedy generations with a 32-token cap.
This is not a complete-answer accuracy dataset.

Quantum labels measure host-to-safe scheduling impact.
Decode labels measure time to the next observed completed output token.
Both labels can contain queue and stream effects.
No future label enters feature extraction.

Observer schemas 1/2/3 coexist.
Missing load/graph features retain explicit flags.
The source hash and version remain in raw records.
Do not rewrite old rows to imply later feature coverage.

Compressed model tables preserve run IDs and source-family hashes.
The prediction export additionally joins each held-out prediction to its raw source SHA256.
Raw JSON archives preserve native snapshots and timestamps.
Large captured tensors remain local with an explicit hash manifest.

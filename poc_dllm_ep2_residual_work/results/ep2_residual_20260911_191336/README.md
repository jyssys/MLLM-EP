# Result inventory

This directory is the result root for the true-EP2 residual-work screen.

Committed compact evidence:

- true-EP2 audit JSON;
- clean/instrumented timing JSON;
- temporal-capture summary JSON;
- aggregate analysis CSV/JSON;
- all required plots.

Large local-only evidence (ignored by Git):

- PyTorch temporal tensor captures (~1.31 GB total);
- profiler Chrome traces;
- per-token and per-branch intermediate CSVs (~21 MB).

The local-only data are reproducible with `scripts/capture_temporal_work.py`.
They are intentionally not placed in Git history.

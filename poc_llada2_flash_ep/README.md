# LLaDA2.0-Flash 100B EP scaling and method PoC

This directory is the isolated workspace for the large sparse dLLM-MoE
substrate study. Raw model weights and external repositories are not committed.
Only reproducible scripts, small measurements, reports, figures, and patches
belong here.

## Final disposition

- Label: `CHARACTERIZATION-ONLY`
- Main report: `../poc_flashvep/reports/llada2_flash_100b_ep_scaling_method_poc.md`
- Compact measurements: CSV/JSON files at this directory root and under
  `results/**/analysis/`
- Raw per-rank trace JSONL streams: retained locally and intentionally ignored
  by Git because they total about 1.2 GB
- Full stdout/stderr logs: retained locally under `logs/` and intentionally
  ignored; compact parsed CSVs and result JSONL remain versioned
- Runtime patch: `patches/dinfer_llada2_true_ep4_and_trace.diff`
- Rebuild final tables/figures: `python scripts/build_final_analysis.py`

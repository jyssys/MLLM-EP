# Final real-DeepEP modality/volume analysis

Analysis outputs are in this directory. The raw runs are adjacent:

- `../modality_aware_tp_ep_real_deepep_20260904_tp_only_sync/`
- `../modality_aware_tp_ep_real_deepep_20260904_real_deepep_v6/`

The raw DeepEP run includes all four GPU worker JSONL files. Its DP1 padding
forwards (8 assignments) remain in raw data and are explicitly excluded by
`analyze_volume.py`; the request-owned DP0 rows are retained for paired
T_MoE comparison.

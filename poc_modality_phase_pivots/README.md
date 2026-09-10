# Modality/phase pivot PoCs

This directory is independent of prior `poc_flashvep` artifacts. It evaluates
three bounded questions without implementing a production scheduler:

1. full-size Attention/DeepEP-stage resource co-scheduling;
2. phase- and layer-granular scheduling oracles;
3. phase-specific DeepEP communication/expert aggregation policies.

All GPU launchers refuse any mapping except physical GPUs 4,5,6,7. CPU
offloading, token-axis splitting, backend switching, and model/routing changes
are excluded.

## Final artifacts

- [Runtime audit](reports/runtime_audit.md)
- [Previous evidence](reports/previous_evidence.md)
- [Track 1: resource co-scheduling](reports/track1_resource_coscheduling.md)
- [Track 2: layer/phase scheduling](reports/track2_layer_phase_scheduling.md)
- [Track 3: EP policy asymmetry](reports/track3_ep_policy_asymmetry.md)
- [Prior-art matrix](reports/prior_art_matrix.md)
- [Final comparison](reports/final_report.md)
- Aggregated result root:
  `results/modality_phase_pivots_20260910_233000/`

Reproduce the aggregation with:

```bash
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_modality_phase_pivots.analyze \
  --root poc_modality_phase_pivots/results/modality_phase_pivots_20260910_233000
```

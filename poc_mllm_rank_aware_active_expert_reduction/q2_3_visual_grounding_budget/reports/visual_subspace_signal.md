# Single-pass visual grounding signal

The frozen score is `log((E_vision+1e-6)/(E_text+1e-6))`, where energies project the current decode pre-MoE hidden onto rank-4 visual/text prefill subspaces from the same layer. At most 16 rows per modality and TP shard were retained.

- Captured condition-sites: 64.
- Held-out visual-score mean/median: -0.5363 / -0.5308.
- Held-out feature/damage Spearman: `{'full_energy': 0.21478260869565213, 'router_removed_mass_control': 0.16173913043478258, 'separated_visual_score': -0.03565217391304347, 'text_energy': 0.23043478260869563}`.
- Strongest control was frozen using calibration only: `full_energy`.

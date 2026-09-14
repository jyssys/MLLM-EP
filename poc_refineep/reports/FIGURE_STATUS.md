# Required figure status

| spec figure | artifact/status |
|---|---|
| fresh M vs refinement | `01_fresh_m_over_refinement.png` |
| fresh remote assignments vs refinement | `18_remote_assignments_over_refinement.png` |
| refinement shape histogram | `02_fresh_m_histogram.png`, `03_fresh_m_cdf.png` |
| existing path latency vs M | `08_existing_path_envelope.png`, `20_controlled_path_envelope.png` |
| existing path latency vs bytes | `19_existing_latency_vs_bytes.png` |
| normal/LL/V2 crossover | `09_existing_winner_mass.png`; V2 blocked by software contract |
| large/medium/small Nsight | raw very-small and large `.nsys-rep`; medium omitted after envelope was decisive |
| runtime tax breakdown | `10_runtime_tax_atlas.png` |
| measured path vs physical lower bound | `10_runtime_tax_atlas.png`, `11_per_shape_o3_headroom.png` |
| request kernel headroom | `12_request_oracles.png`, `13_mode_switch_vs_third_path.png` |
| RefineEP by M/skew/fanout | not applicable: CUDA gate failed; controlled existing-path envelope retained |
| end-to-end BCT | analytical-only `12_request_oracles.png`; no prototype result |
| kernel mechanism ablations | `16_control_floor_sensitivity.png`; no kernel mechanisms implemented |
| fraction invoking RefineEP | not applicable: no RefineEP path |

Missing prototype-only figures are intentionally not fabricated after the
mandatory CUDA gate failed.

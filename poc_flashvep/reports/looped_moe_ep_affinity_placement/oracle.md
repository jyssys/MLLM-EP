# Placement and actionable-headroom analysis

IN_PROGRESS: this is a preview, not a completed PoC.

The detailed table is `placement_oracles.csv`.
It contains every model, EP degree, batch, and required prefix length.

| Oracle | Available evidence | Missing evidence |
|---|---|---|
| O0 | One destination per token is an unconstrained fanout lower bound. | A legal complete-model wall-time schedule. |
| O1 | Calibration-only generic fixed maps preserve expert counts and HBM. | A validated dispatch/GEMM/layout E2E cost model for these maps. |
| O2 | Transition-augmented maps use the same fixed-memory constraints. | An additional real state-transfer edge removed by recurrent information. |
| O3 | Held-out route prediction measures information, not perfect foresight. | A legal online policy with future exact routing. |
| O4 | Separate live static and current-router generic controls exist. | A validated oracle above strong production generic EP. |

The maps use finite pair swaps, not a proven global optimum.
Each physical expert has one fixed owner.
Every rank preserves its original expert count and BF16 weight capacity.
Changed initialization bytes appear in `runs/fixed_placement_maps.json`.
They are not measured transfer latency.
No per-loop weight remapping receives zero cost.

Destination fanout, payload, and rank-load dispersion are mechanism metrics.
They must not become E2E milliseconds without a validated cost model.
The original dispatcher sends one activation for each selected expert.
Co-location does not itself deduplicate that payload.
Destination deduplication also overlaps existing generic MoE work.

Static-to-generic measured ratios appear in `generic_control.csv`.
Generic-to-loop and static-to-loop live ratios remain NOT_AVAILABLE because no loop-aware runtime passed the gate.
This table does not silently replace missing speedups with 1.000×.
The 1.15× generic-relative prototype gate is not established.

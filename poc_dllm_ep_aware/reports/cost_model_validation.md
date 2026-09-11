# EP4 cost-model validation

## Calibration data

The calibration uses the exact LLaDA expert dimensions (H=2048, I=1024),
BF16, 16 experts resident on each of four H100s, NCCL assignment A2A, and the
vLLM fused-expert kernel.  The 84 configurations span:

- 256, 1,024, and 4,096 assignments;
- destination fanout 1–4;
- remote fraction 0.25/0.5/0.75 when feasible;
- 2/8/16 active experts per active rank;
- five warmups and 30 measured repetitions.

The split holds out entire route configurations rather than random repetitions
of seen shapes.

## Models

| Model | Features | RMSE | MAPE | R² |
|---|---|---:|---:|---:|
| C0 | bytes | 0.0521 ms | 4.99% | -0.061 |
| C1 | bytes + fanout | 0.0521 ms | 4.99% | -0.061 |
| C2 | calibrated dispatch + expert + combine features | 0.0495 ms | 4.68% | 0.039 |
| C3 | C2 + critical-rank/fragmentation features | **0.0491 ms** | **4.44%** | **0.058** |

## Interpretation

The small percentage error is misleading: the surface is dominated by an
approximately 0.61 ms startup floor.  Assignment count changes by 16× while
median total operator time changes by about 4%.  C3 marginally reduces error,
but its near-zero R² means it cannot reliably choose between nearby actions.
Adding fanout provides no held-out gain on this reference collective.

The cost model is therefore adequate for conservative ceilings and rejecting
linear `pair reduction = latency reduction` assumptions.  It is not adequate
as a runtime decision model and was not used to manufacture a positive policy.

## Generality boundary

This is an NCCL reference assignment-A2A surface, not DeepEP.  Absolute startup
and fanout behavior must be recalibrated for DeepEP, PPLX, or EP8.  Because all
candidate request-level oracles fail before that step, production-backend
calibration would not change the implementation decision.

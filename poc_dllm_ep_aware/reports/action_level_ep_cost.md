# Action-level EP cost and calibrated surface

## Native action units

| Family | Native utility proxy | Physical action measured | Main finding |
|---|---|---|---|
| TEAM | NFE reduction / speculative acceptance | optional speculative branch rows | 21.4% fewer EP4 calls but 32.1% more assignments and bytes |
| REFLEX | refinement role and allocated k | each marginal expert branch | AvgK −6.05% gives assignment-aware remote rows −7.07%, but stock naive bytes do not change |
| DES | expert vote / 38-expert coreset | inclusion of one expert in the coreset | unique support shrinks but top-8 branch count and fanout remain unchanged |

The action-level features retained for candidate analysis are remote assignment
count/fraction, destination ranks, dispatch/combine bytes, critical-rank rows,
active local experts, and active-experts-per-critical-row fragmentation.

## EP4 cost surface

The 84-shape, 30-repetition surface used H=2048/I=1024 BF16 fused experts and
reference NCCL A2A.  Median end-to-end operator time is strongly startup
dominated in this range:

| Axis | Regime | Median total |
|---|---|---:|
| Assignments | 256 | 0.6127 ms |
| | 1,024 | 0.6089 ms |
| | 4,096 | 0.6384 ms |
| Fanout | 1 | 0.6133 ms |
| | 2 | 0.6174 ms |
| | 3 | 0.6306 ms |
| | 4 | 0.6024 ms |

The non-monotonic fanout-4 result is a reference-NCCL collective effect, not a
general DeepEP law.  The economic fact is robust: changing assignment count by
16× changes median operator time by only about 4% at these shapes.  This is why
assignment reduction cannot be mapped linearly to request speedup.

## Held-out cost models

Entire route configurations were held out.  Lower MAPE reflects the flat
startup floor; low R² shows that these models do not rank the small residual
variation reliably.

| Model | Held-out RMSE | MAPE | R² |
|---|---:|---:|---:|
| bytes only | 0.0521 ms | 4.99% | −0.061 |
| bytes + fanout | 0.0521 ms | 4.99% | −0.061 |
| dispatch+expert+combine features | 0.0495 ms | 4.68% | 0.039 |
| critical-rank model | **0.0491 ms** | **4.44%** | **0.058** |

Adding fanout to bytes gives no held-out improvement.  Critical-rank features
are marginally best, but the explanatory signal is too weak to justify making
the predictor a contribution or using it to drive a controller.

## Stock-backend mismatch

The local vLLM `NaiveAll2AllManager` dispatch multicasts full hidden and router
logits, then combines with a full-hidden all-reduce.  Therefore selected-pair
count is not even in its communication byte equation.  A paper method would
first require a production ragged assignment transport; that is ordinary
backend enablement, not evidence that an EP-aware semantic policy has large
headroom.

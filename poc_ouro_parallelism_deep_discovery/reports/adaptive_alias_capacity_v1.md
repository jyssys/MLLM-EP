# Adaptive terminal-alias payload screen

## Question and model

The [C03 scope](negative_scope_audit_v1.md) must not imply zero adaptive capacity opportunity.
This CPU supplement uses the128 admitted Adaptive request traces.
It does not implement aliases or change any gate.

Let P be full-R4 prompt positions.
Let d_i be the observed exit depth for each recorded decode position.
Let k be BF16 K+V bytes for one position across all physical layers at one depth.
Then the declared committed-payload model is:

```text
FullDepth = 4 * (P + number_of_recorded_decode_positions) * k
IdealAliases = (4 * P + sum(d_i)) * k
```

Here k is196608 bytes for Base1.4B and393216 bytes for Base2.6B.
The model preserves every computed depth state.
It aliases only terminal values that `last_exited` otherwise copies into deeper slots.
FixedR4 has zero saving under this specific rule.

## Structural results

The mean below weights requests equally.
Each confidence interval uses4000 request-bootstrap resamples.
The aggregate column instead divides summed modeled payload savings by summed full-depth payload.

| Base | Domain | N | Mean request payload saving | 95% CI | Aggregate payload saving |
|---|---|---:|---:|---|---:|
| 1.4B | GSM8K | 32 | 5.478% | [5.085, 5.883]% | 5.601% |
| 1.4B | MATH-500 | 32 | 23.340% | [21.754, 24.913]% | 28.412% |
| 2.6B | GSM8K | 32 | 5.802% | [5.416, 6.191]% | 5.922% |
| 2.6B | MATH-500 | 32 | 25.336% | [24.687, 26.022]% | 25.969% |

The full-R4 prompt dilutes terminal-alias savings in short GSM8K outputs.
Longer outputs can expose a larger committed-payload fraction.
This is a capacity observation within the model, not measured admission or speedup.

## Physical obligations remain

The inspected CDB code copies rows into deeper cache tensors.
It performs copies before deeper reads and assumes one recurrent token per request in flight.
An alias runtime must preserve this publication order and every state lifetime.
The [source receipt](../artifacts/adaptive_alias_capacity_source_v1.json) pins the relevant paths.

The model excludes page granularity, alias metadata, lookup, static graph allocation, transient writes, allocator slack, weights, and scratch.
It neither measures allocated/reserved HBM nor infers HBM traffic.
Its request totals are not a simultaneous-concurrency memory peak.
The [raw modeled rows](../analysis/adaptive_alias_capacity_v2/requests.csv) retain the denominator for every request.

### Whole-page restriction

The [v2 sensitivity](../analysis/adaptive_alias_capacity_v2/summary.json) charges a rounded page denominator.
It aliases only sealed pages whose rows all select the same shallower source depth.
It never aliases the last partial page.
This is a narrower mechanism than ideal row aliases, not a bound against finer mappings.

| Base/domain | Page16 | Page32 | Page256 |
|---|---:|---:|---:|
| 1.4B GSM8K | 1.472% | 0.323% | 0.000% |
| 1.4B MATH-500 | 1.500% | 0.189% | 0.000% |
| 2.6B GSM8K | 3.178% | 1.839% | 0.000% |
| 2.6B MATH-500 | 6.599% | 3.043% | 0.000% |

These values are mean request payload savings under that page model.
The inspected source default is16 positions per page.
This supplement performs no new GPU allocator inspection.
Page32 and page256 are modeled sensitivities, not executed configurations.
The legacy cache still owns separate depth tensors.
An actual alias implementation needs compatible indirection, safe reclamation, and publication.
The baseline constructor requests640 blocks.
This payload model does not reclaim that configured static reservation.

This positive structural result changes no candidate selection.
Terminal aliases overlap existing adaptive-KV work.
No recurrence-specific same-P benefit, executable allocator, numerical runtime, or quality result follows.
Six CPU tests check the row formula, page restriction, and guards.

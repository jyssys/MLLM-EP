# Cross-method characterization

## Main table

Raw latencies are comparable only within each model/method's own vanilla
control.  Parenthesized REFLEX/DES speedups are diagnostic and invalid because
their bounded quality gate failed.

| Method | Model | Setting | Quality status | Speedup vs own vanilla | Algorithmic proxy | Expert-pairs | Remote assignments/bytes | Fanout |
|---|---|---|---|---:|---:|---:|---:|---:|
| TEAM | SDAR | single | bounded positive | **1.832×** | NFE −41.7% | +11.5% | N/A | N/A |
| TEAM | SDAR | EP2 | trajectory preserved | **1.179×** | NFE −33.3% | +31.5% | +33.6% | 2→2 |
| TEAM | SDAR | EP4 | prefix-only quality | **0.821×** | NFE −21.4% | +32.1% | +32.1% | 4→4 |
| REFLEX port | LLaDA | single | **invalid, 0/4 vs 3/4** | (4.36×) | AvgK −6.05% | −6.05% | N/A | N/A |
| REFLEX structural | LLaDA | EP4 | **invalid** | excluded | AvgK −6.05% | −6.05% | −7.07%/row | 4→4 |
| DES port | LLaDA | single | **invalid, 1/4 vs 3/4** | (2.05×) | active coreset −40.6% | 0% (top-8) | ≈0%/row | 4→4 |

## What is causally established

TEAM provides a controlled within-method result: fewer denoising forwards can
coexist with more token-expert assignments and more network payload, and the
winner reverses as physical EP degree increases.  The released algorithmic
proxy therefore does not predict physical distributed cost.

REFLEX and DES add two structural lessons, not two additional method failures:

- a smaller per-token expert budget reduces bytes only on an assignment-aware
  backend; stock naive EP sends full hidden/router tensors regardless of k;
- a much smaller global coreset can leave physical rank fanout unchanged when
  top-k is high relative to experts per rank.

Because their official execution contracts were unavailable and the ports did
not preserve benchmark quality, these two lessons do not satisfy the spec's
multi-method motivation gate for a general paper method.

## Motivation gate

- M1 (two quality-valid methods mismatch): **FAIL**.
- M2 (valid cross-method differential explained by EP cost): **FAIL**; only
  TEAM is faithful end-to-end.
- M3 (quality-valid method plus >=10% EP-aware oracle): **FAIL**; the strongest
  future-aware bound is 6.28%.

The correct outcome is a strong characterization of TEAM's EP4 reversal, not a
universal training-free EP-aware method claim.

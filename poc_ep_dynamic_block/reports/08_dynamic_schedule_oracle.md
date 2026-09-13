# Actual-trajectory dynamic schedule oracle

## Strict exhaustive result

For each task, all 56 ordered compositions of 128 using B in
`{16,32,64,128}` were executed on the real model trajectory at n=8/mini8. Four
are fixed controls and 52 are genuinely mixed. No independent block timing is
summed.

| Task | fastest fixed | time / score | fastest mixed | time / score | mixed gain |
|---|---|---:|---|---:|---:|
| GSM8K | 32-32-32-32 | 8.200 s / 5-of-8 | 32-32-32-16-16 | 10.596 s / 5-of-8 | -29.22% |
| HumanEval | 128 | 7.576 s / 3-of-8 | 32-32-16-32-16 | 11.447 s / 3-of-8 | -51.10% |

O0 latency-only, O1 dataset-score-preserving, O2 per-request-safe, and O3
one-sample-epsilon all select the same fastest mixed row in both exhaustive
runs and are negative by the margins above. Mixed schedules can gain one
bounded success, but they are slower: the fastest score-6 GSM8K mixed schedule
takes 16.518 s, and the fastest score-4 HumanEval mixed schedule takes 11.717 s.

## Global fixed frontier

The same conclusion holds against all matching fixed schedule controls and the
larger n=32 fixed-B/mini/threshold frontier. Dynamic schedules are never credited
against a weak fixed endpoint bundled into their own run.

## Future-aware request oracle

At mini=1, every request is independently evaluated under five fixed and seven
mixed schedules. This unrealistically provides future trajectory information
and removes cross-request waiting. Even then, median safe gain is only 1.014%
on GSM8K and 0.369% on HumanEval. Two GSM8K requests have 9.15--10.26% local
headroom, but the aggregate mean is 3.171% and the median fails the 5% gate.

## Why mixed B loses

Changing width changes the denoising trajectory and often increases NFE. In a
batch, sequences also complete blocks at different iterations, creating
width-specific sub-waves and worse physical packing. A global schedule barrier
trades fragmentation for waiting and does not restore the fixed frontier. The
cost is therefore structural to this current decoder/runtime contract, not a
controller-selection error.

## Gate

The quality-safe actual-trajectory oracle is non-positive in both exhaustive
tasks and below 5% even in the maximally favorable per-request tournament.
Dynamic-controller implementation is forbidden by the working contract.

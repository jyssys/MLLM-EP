# Final decision

## Label

`CHARACTERIZATION-SIGNAL`

## Chain verdict

| Required link | Verdict | Evidence |
|---|---|---|
| B / AR-ness changes physical sparse-EP shape | PASS | B8→B128 changes M 48→704, rows/active-expert 3.67→29.76, tiny-expert share 75.4%→29.3%, and BF16 dispatch bytes 2.36→34.90 MB in the controlled GSM8K trace; HumanEval agrees. |
| Physical regime survives controls | PARTIAL PASS | Matched M preserves a B128 context/scheduling penalty, but B16--64 per-wave walls converge within about 7%; NFE remains trajectory-dependent even with threshold 1.0 and EOS early stop disabled. |
| Requests/states have different safe optima | PASS, SMALL | Fixed-B preference varies across requests, but future-aware median safe mixed gain is only 1.014% GSM8K and 0.369% HumanEval. |
| Actual dynamic trajectory beats strongest fixed frontier | FAIL | Exhaustive mixed trajectory is 29.22% slower on GSM8K and 51.10% slower on HumanEval at the same bounded score. |
| EP features add useful trigger value | FAIL / NOT TRIGGERED | They predict physical wave cost, but the target oracle is already below 5% and semantic NFE/quality dominates the safe request winner. |
| Live method gate | FAIL | No controller is justified or implemented. |

## Decisive actual-trajectory oracle

All 56 ordered compositions of 128 from B={16,32,64,128} were run per task at
n=8/mini8. GSM8K's fastest fixed path is 8.200 s versus 10.596 s for the fastest
mixed path, both 5/8. HumanEval's fastest fixed path is 7.576 s versus 11.447 s
for the fastest mixed path, both 3/8. The latency-only, dataset-score,
per-request-safe, and one-sample-epsilon gates all remain negative.

## Why the promising premise does not become a method

Block size is a strong physical control knob, but changing it also changes the
refinement trajectory. Within a batch, sequences cross block boundaries at
different times, causing width-specific sub-waves, lower fill, and often more
NFE. A barrier replaces fragmentation with waiting. Best-per-B mini calibration
already captures most of the static physical advantage. The remaining
future-aware per-request opportunity is too small to fund controller overhead or
to clear the paper-level headroom gate.

## Boundaries

- This does not show that adaptive B can never help another checkpoint or a
  decoder trained explicitly for variable schedules.
- It does show that on this downloaded LLaDA2.0-Flash 100B checkpoint and true
  EP4 runtime, dynamic B does not beat the fixed Pareto frontier under the tested
  exact block-boundary semantics.
- Dead-row savings are not claimed; they belong to Epoch-like compaction.
- No online-serving, TP4, or production-controller result is claimed.

## Recommendation

Do not continue RAWS/dynamic-B implementation on this substrate. Retain the
variable-B harness and physical regime atlas as reusable characterization. A
future revisit should require a materially different decoder contract that
prevents mixed-width NFE inflation and preserves large physical waves; retuning
the current controller cannot recover the measured negative margins.

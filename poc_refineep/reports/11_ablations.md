# Ablations and robustness

## Restart robustness

Five independent process restarts each evaluate all 25 real cases. LL beats
normal in every case/restart. Median paired LL gains by restart are 62.34%,
57.42%, 62.68%, 58.65%, and 62.07%. This invariance is stronger evidence than
the noisier unpaired absolute medians.

## Route geometry

The 15 controlled cases vary M, fanout, remote fraction, and rank skew. LL wins
15/15. The smallest advantage, 18.0%, occurs for M512 with strong rank skew;
the largest, 63.9%, occurs at M32. Therefore the real-route result is not
explained by one fanout or balanced-load artifact.

## Handle reuse

Normal cached-handle reuse removes about 40--46 us/wave. It remains slower
than LL in 23/25 real cases. It only wins at M802/1024, where a changing
refinement route generally cannot reuse the exact handle. This attacks P1
without overstating an invalid future-known control.

## Control-floor sensitivity

| total two-direction control | GSM8K extra vs O0 | HumanEval extra vs O0 |
|---:|---:|---:|
| 25 us | 3.34% | 3.25% |
| 50 us (O3) | 2.29% | 2.10% |
| 75 us | 1.23% | 0.95% |
| 100 us | 0.45% | 0.23% |
| 125 us | 0.11% | 0.06% |

O1 eliminates control entirely and still gives only 4.40%/4.39%. Hence the
negative is insensitive to the exact credible-control constant.

## Instrumentation boundary

CUDA-event medians are primary. Nsight Systems is secondary and includes
initialization outliers; its total duration is excluded. Nsight Compute was
unavailable. No observer-heavy number is used as a request speedup.

# Future-free dormant classifier

A simple logistic proxy was trained on sequence IDs 0--23 and tested on the
held-out last quarter.  It uses layer-16 traces and chooses the lowest training
threshold with at least 99% dormant precision.  False-dormant rate is the
primary safety metric.

| task/features | test false-dormant | dormant recall | perfect-oracle recovery | projected post-Epoch E2E |
|---|---:|---:|---:|---:|
| GSM semantic | 0% | 33.80% | 33.80% | 2.350% |
| GSM expert only | 0% | 0.054% | 0.054% | 0.004% |
| GSM semantic+expert | 0% | 34.04% | 34.04% | 2.367% |
| Human semantic | 0% | 18.91% | 18.91% | 1.375% |
| Human expert only | 0% | 0% | 0% | 0% |
| Human semantic+expert | 0% | 17.87% | 17.87% | 1.300% |

At a deliberately high-safety threshold the semantic proxy recovers only a
minority of the impossible future-aware ceiling.  Expert/EP features add
0.24 percentage points recall and 0.017% E2E on GSM8K, and hurt HumanEval.
They do not supply the hoped-for independent signal.

This is a single-batch, sequence-held-out diagnostic rather than a broadly
generalized predictor.  Its best possible economic result is already below the
implementation gate, so a larger learned controller is unwarranted.

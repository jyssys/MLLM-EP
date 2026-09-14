# In-flight concurrency

The harness respects legacy LL's documented two-buffer lifetime: no third
receive tensor is retained before a buffer slot becomes reusable. `ll_two_slot`
issues two dispatches asynchronously, chains their combines, synchronizes, and
then advances to the next legal pair. All outputs were checked.

For real LLaDA2 routes offset across refinement ages, three-restart medians are:

| Q | LL serial waves/s | LL legal-two-slot waves/s | Normal waves/s |
|---:|---:|---:|---:|
| 1 | 7,024 | 6,881 | 3,296 |
| 2 | 7,730 | 8,228 | 3,482 |
| 4 | 9,321 | 10,237 | 3,947 |
| 8 | 10,191 | 11,569 | 4,189 |
| 16 | 10,346 | 11,705 | 4,268 |

At Q=16, legal two-slot LL is 13.1% faster than serial LL and 2.74x Normal.
Q=8 to Q=16 still improves 1.17%; it does not collapse. The max real-route
relative L2 was 7.71e-4, consistent with BF16 reduction-order drift.

Conclusion: the two-buffer contract constrains object lifetime, but it did not
serialize the measured serving stream into a throughput failure. See
`INFLIGHT_RESULTS.csv` and Figures 05--07.

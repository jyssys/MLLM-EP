# Mini-batch and matched-M controls

Mini size is a first-order confound, so dynamic B is compared against each B's
strongest feasible mini point rather than a shared weak default.

## Matched-M replay

For GSM8K at configured M=256, median actual M and forward wall are:

| B / mini | actual M | forward wall |
|---|---:|---:|
| 8 / 32 | 208 | 122.59 ms |
| 16 / 16 | 256 | 117.83 ms |
| 32 / 8 | 256 | 125.59 ms |
| 64 / 4 | 256 | 122.01 ms |
| 128 / 2 | 256 | 172.75 ms |

B16/B32/B64 converge within roughly 7% per wave. B128 remains slower because
attention/context and scheduling work grow even when routed M is held fixed.

The full-request cost does not converge: smaller mini sizes create far more
physical waves (133/148/256/503/1114 across the table). Therefore the apparent
static B advantage is largely a packing/NFE effect, and a dynamic policy must
pay real width-fragmentation and wave-count costs.

## Prior RAWS control

The preceding same-substrate RAWS study found a ready-set-feasible dynamic
mini-size oracle of 0.650%, and only 0.224% after hypothetical live-row
compaction. Those are contextual controls, not measurements from this campaign;
they independently weaken the idea that a hidden mini-size controller can
rescue mixed-B fragmentation.

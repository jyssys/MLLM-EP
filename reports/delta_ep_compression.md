# DeltaEP lossless compression

Discovery IDs 0–7 chose and froze a byte bitmap + changed-byte literals + one-byte mode tag with full-vector fallback. The table is the one-shot held-out result on IDs 8–15.

| Boundary | Predictor | median ratio | aggregate ratio | fallback | zero XOR bytes |
|---|---|---|---|---|---|
| dispatch | adjacent | 1.191× | 1.204× | 0.24% | 28.56% |
| dispatch | wrong_token | 1.117× | 1.138× | 4.13% | 22.97% |
| dispatch | nonadjacent | 1.140× | 1.157× | 0.42% | 24.80% |
| dispatch | zero_predictor | 1.000× | 1.000× | 100.00% | 0.29% |
| combine | adjacent | 1.112× | 1.134× | 10.19% | 22.63% |
| combine | wrong_token | 1.045× | 1.076× | 24.35% | 16.80% |
| combine | nonadjacent | 1.070× | 1.094× | 16.09% | 19.07% |
| combine | zero_predictor | 1.000× | 1.000× | 100.00% | 0.20% |

The word-bitmap codec falls back on the median (1.0×). CPU zstd/lz4 are not used as proposed codecs, and no CUDA codec was built because the lossless gate already fails.

## Predefined layer stratum

| Layer | discovery median | held-out median | held-out aggregate |
|---|---|---|---|
| 1 | 1.375× | 1.369× | 1.330× |
| 5 | 1.153× | 1.141× | 1.161× |
| 10 | 1.129× | 1.122× | 1.141× |
| 14 | 1.168× | 1.165× | 1.174× |
| 19 | 1.258× | 1.238× | 1.230× |

Discovery selects layer 1; its frozen held-out median is 1.369×. The subgroup reproduces, but five-layer and all-layer stage upper bounds remain far below the systems gate, so it is not promoted.

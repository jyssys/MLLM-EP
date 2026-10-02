# Expert Footprint Analysis

Measured clean TRUE EP4 on GPUs2,3,6,7, output cap64/EOS, first16 frozen requests (8+8). Route features come from a separate trace pass with identical request/prefix/draft IDs.

| Arm | acceptance | progress/verify | experts/layer | remote paths/layer | fanout/token | max-rank bytes/layer | EP ms/progress | target ms/progress | E2E ms/token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AR | N/A | 1.000 | 8.000 | 5.932 | 3.644 | 14924.6 | 25.953 | 90.804 | 92.779 |
| 2 | 0.858 | 2.673 | 17.123 | 17.846 | 3.642 | 29840.6 | 9.434 | 33.214 | 46.071 |
| 4 | 0.742 | 3.850 | 23.738 | 29.714 | 3.641 | 44742.9 | 7.140 | 23.832 | 41.043 |
| 8 | 0.626 | 5.712 | 33.822 | 53.530 | 3.643 | 74636.0 | 4.574 | 15.733 | 37.961 |

Expert counts are unique global expert IDs per layer, averaged over48 layers. Expert union sums layer/expert pairs, not IDs across layers. Actual loaded tensors imply9,437,184 bytes/expert; multiplying by active experts is a working-set proxy, not measured HBM traffic. Actual dummy/padded expert counts remain in the M1 baseline.

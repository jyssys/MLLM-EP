# Large-M crossover

We measured global M from 1 through 8192 with uniform, real-like, mild-skew,
and strong-skew routes. Each source rank carried an even share of M. The timing
unit is the maximum same-repeat rank transaction wall, then the median over
repetitions.

| Global M | LL real-like (ms) | Normal real-like (ms) | LL speedup |
|---:|---:|---:|---:|
| 1,024 | 0.173 | 0.310 | 1.79x |
| 2,048 | 0.317 | 0.445 | 1.41x |
| 4,096 | 0.565 | 0.611 | 1.08x |
| 8,192 | 0.930 | 0.778 | 0.84x |

Normal wins all four route shapes at M=8192; LL wins all 52 route shapes at
M<=4096. The crossover is therefore between 4096 and 8192. It is outside the
measured LLaDA2 compacted range (maximum 1024), so it does not create a useful
per-wave switch in the target serving workload.

M=16384 is `ENVIRONMENT_BLOCKED`, not a loss: legacy V1's requested LL RDMA
heap violates an internal 32-bit indexing assertion. The stress endpoint is
excluded from winner counts. See `M_CROSSOVER.csv` and Figures 01--02.

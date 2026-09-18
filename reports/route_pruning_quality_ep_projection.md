# Route-pruning EP4/EP8 projection

These numbers are routed-MoE-stage projections, **not request E2E latency**.
EP4 and EP8 are labeled SIMULATED-EP4/8-EP2-CALIBRATED. Dispatch and combine
use the true-EP2-calibrated communication model; owner-local expert compute
uses actual GPU 0/1 grouped-mm replay, extended with 128 representative
route-pruned shapes per EP degree. The evaluated assignment ranges are fully
inside the resulting calibration envelope.

| EP | Method | Dispatch ms/req | Expert ms/req | Combine ms/req | Stage ms/req | Total stage gain | NFE-normalized gain | Max/mean | Wait |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | Vanilla | 310.642 | 1532.745 | 171.725 | 2015.112 | 0.000% | 0.000% | 1.1058 | 9.341% |
| 4 | P2 1% | 260.761 | 1277.979 | 143.957 | 1682.696 | 16.496% | 1.487% | 1.1037 | 9.149% |
| 4 | P4 1% | 263.268 | 1284.098 | 145.330 | 1692.697 | 16.000% | 1.896% | 1.0958 | 8.510% |
| 8 | Vanilla | 307.837 | 1258.864 | 169.961 | 1736.661 | 0.000% | 0.000% | 1.2122 | 17.017% |
| 8 | P2 1% | 258.245 | 1056.814 | 142.375 | 1457.434 | 16.078% | 0.995% | 1.2178 | 17.376% |
| 8 | P4 1% | 260.286 | 1059.866 | 143.455 | 1463.607 | 15.723% | 1.572% | 1.2029 | 16.398% |

## P4 gain decomposition

P4's total projected routed-MoE-stage reductions are
16.000% at EP4 and
15.723% at EP8. They cannot be called a
quality-safe latency gain. Mean NFE fell 14.377%,
which explains most of both numbers. After normalizing stage cost by NFE, the
remaining gains are only 1.896%
(EP4) and 1.572% (EP8).

At EP8, P4 does improve physical shape relative to Vanilla: max/mean changes
from 1.2122 to 1.2029, and wait fraction
from 17.017% to 16.398%.
But P2's total EP8 stage reduction is larger (16.078%)
because its trajectory happened to use fewer NFEs; P4 therefore has no positive
total-gain increment over P2. On the NFE-normalized view P4 exceeds P2 by only
0.578
percentage points, far below the 3-point EP-specific gate.

Remote-byte reductions include the changed trajectories and are not isolated
route-only savings. P4 EP8 remote logical bytes/request change from
40.150 to
33.153 GiB.

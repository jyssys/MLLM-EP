# Block-hot exact expert replica cache oracle

This candidate keeps exact expert weights and asks whether copying a remote
expert once per diffusion block amortizes over temporally persistent demand.
The conservative feasible variant replicates to the source rank, charges
measured P2P copy cost and HBM, preserves expert compute, and adds a
critical-rank-load penalty when relocated work overloads the source. The
perfect-future communication-coverage estimate remains an oracle, not live
latency.


The exact SDAR expert is 9.437 MB in FP16. GPU0→GPU1 P2P copy medians were
0.0619 ms for one expert and 0.2311 ms for eight; concurrent useful compute
left 0.0357 ms and 0.1638 ms visible respectively. Copy bandwidth is therefore
not the bottleneck in the oracle.

Perfect-future per-block caching nevertheless fails economically. One, two,
four, and eight replicas per block produce optimistic net clean-request gains
of 0.070%, 0.110%, 0.153%, and 0.133%. At larger budgets the extra work placed
on the source/replica rank exceeds the proportional communication saving and
net gain becomes zero. The best point (four replicas) saves 224.4 ms of scaled
communication across the captured requests, but adds 150.5 ms of
critical-rank expert work plus 2.48 ms of copies.

**Decision: KILL (<5%).** The small remaining value is also directly adjacent
to predictive expert replication and does not justify an implementation.

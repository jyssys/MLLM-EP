# Best-existing backend oracle

O0 selects the faster measured Normal or LL service time for every ready wave.
Because all real compacted M values are <=1024 and LL wins throughout that
range, O0 exactly equals LL-only:

- throughput gain: 0.00%;
- p50/p95/p99 reduction: 0.00%;
- selected Normal waves: 0/146 chronological shapes.

A threshold switch at global M≈6144 reproduces the same result. Consequently a
Normal/LL selector is neither useful nor novel for this substrate. Figure 14
shows the zero oracle.

# Candidate kernel contract

No RefineStreamEP CUDA kernel was implemented. The mandatory promotion gate
failed before implementation:

- measured existing LL remains the best real-shape backend;
- legal two-slot execution improves rather than harms Q=16 throughput;
- capacity mismatch is a memory issue without a monotonic latency tax;
- credible combined serving headroom is <8% tail and <3% throughput.

An exact future design could use transaction IDs, an N-way buffer ring, and
elastic slots. However, those mechanisms are not justified by this substrate
and overlap current DeepEP V2's unified ElasticBuffer direction. Implementing
them here would optimize an unobserved bottleneck.

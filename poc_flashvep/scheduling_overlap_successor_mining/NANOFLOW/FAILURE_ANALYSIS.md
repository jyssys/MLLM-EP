# NanoFlow failure matrix interpretation

1. Activation copy changed sigmoid to SiLU: actual factory CPU regression and
   fresh native HF outputs identify a compatibility bug. A one-argument repair
   fixes it. This is not a successor research opportunity.
2. Plan-unlisted prefill attention was uninitialized: lifecycle compatibility
   repair, then fresh pure-prefill smoke and 18-engine comparison pass. Failed
   startup rows are excluded, not treated as poor method performance.
3. More measured overlap did not imply lower latency: split2 overlaps ~1.63 ms
   of resident NCCL/compute but increases profiled median B16 step from 9.26 to
   12.70 ms, with more kernels and longer communication residency. These are
   separate profiled diagnostics, not clean E2E data. Small-batch overhead is
   already known, and our manual plan is not the searched paper baseline.
4. Large-prefill three-plan clean E2E envelope is zero: plain wins median costs
   at M4096 and M8192. This closes the measured finite portfolio only.
5. Prompt-wide final projection is live-verified by [8192,151936] logits before
   selecting four final positions. Last-token projection is an obvious
   engineering control, not a novel scheduling failure. No savings are assigned
   without timing; it limits fidelity to an optimized paper prefill path.
6. Long-output discrepancies are near ties in independent same-prefix HF
   diagnostics, including same-plan restarts. This weakens a semantic-bug claim,
   but does not certify benchmark quality. Unequal-output performance pairs
   remain excluded from positive evidence.

No material, non-trivial MLLM-specific failure is established. Missing native
MoE search/shape-transition support remains a port boundary, not method failure.

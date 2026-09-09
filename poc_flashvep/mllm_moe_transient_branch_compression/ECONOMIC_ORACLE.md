# Economic oracle and stop boundary

This analysis maps row removal to request latency conservatively.  It does not
call a branch-count reduction a speedup, and it does not sum per-rank timings.

## Measured serving envelope used for the upper bound

The prior clean Qwen3-VL TP2/DP2/EP4 online atlas measured normal layer-local
`T_MoE` medians of 1.16--1.40 ms in the stable text/vision regimes.  Forty-eight
MoE layers therefore account for roughly 56--67 ms against 443--970 ms request
p50, or **6--15% before overlap**; realistic text/vision regimes were 6--10%.
For the high-resolution vision c8 prefill specifically, `T_MoE` was 1.218 ms,
the expert share was 42.25%, and request p50 was 758 ms.  The corresponding
upper bounds are:

- eliminate the entire measured MoE span: `48*1.218/758 = 7.71%` request E2E;
- eliminate all expert compute while dispatch/combine remain: `7.71%*42.25% = 3.26%`;
- eliminate only a down projection (one third of expert GEMM FLOPs): at most
  about 1.09%, even before map/reconstruction overhead;
- eliminate gate/up only (two thirds): at most about 2.17%.

These are deliberately optimistic.  They do not charge compression-map,
representative construction, gather/scatter, expansion, metadata, padding, or
the loss of grouped-GEMM efficiency.

## O1/O2/O3 definitions

- **O1 compute-only:** scale expert time linearly by the quality-safe branch-row
  reduction.  This assumes perfect grouped-GEMM scaling.
- **O2 communication-aware:** additionally allow dispatch/combine rows to scale
  by that fraction.  This is more optimistic than DeepEP's real fixed/notify
  costs.
- **O3 feasible:** O2 minus map, gather, compact-list formation, expansion,
  metadata, and alignment costs.  No prototype is allowed unless O3 reaches
  8% direct request E2E.

The completed nine-sample full-output oracle has median quality-safe row
reduction 0% at every group cap and a maximum single sample/layer reduction of
0.0283%. Thus aggregate O1/O2 round to 0%; O3 cannot be positive. No
compute-only sub-expert child can reach the 8% implementation gate because
even perfect removal of *all* expert compute is only 3.26% in the measured
vision regime.

## Adjacent-child kill gates

1. **Sub-expert sharing:** reusing a gated intermediate `z_j` for token `i`
   produces `W_down z_j = E_e(h_j)`, exactly the already-tested full-output
   substitution.  It cannot improve final error.  Sharing only down work has a
   1.09% perfect request upper bound; sharing only gate/up has 2.17%.
2. **Contribution-weighted selection:** at a 5% row budget, the captured-output
   sharing oracle and zero-output skipping control produce nearly the same
   affected-token error.  Its maximum ideal E2E saving is only 0.16% if only
   expert work scales (`5%*3.26%`), before overhead.  This is also directly
   crowded by MoDES, AnyExperts, and ACE.
3. **Route-run packing:** the full-serving Nsight atlas observed router and
   DeepEP-layout kernels totaling 49.9 ms over a roughly 4.3 s trace.  Even
   deleting both whole categories gives a non-critical-path aggregate upper
   bound of about 1.16%; exact packing can remove only a subset.
4. **Expert codebook/low rank:** a branch-output codebook that magically removes
   all expert execution is still capped by the 3.26% expert share.  It cannot
   pass the 10% child gate on this four-GPU request regime.

## Decision implication

The economic gate is already stricter than the geometry gate for every
compute-only child.  Only a quality-safe full-branch policy could also reduce
communication rows, but it requires at least 15% strict-safe rows and the
completed live geometry reports 0% in the median. A compact DeepEP prototype would therefore be
implementation work after both prerequisite gates failed and is intentionally
not built.

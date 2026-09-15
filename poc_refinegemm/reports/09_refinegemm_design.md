# RefineGEMM design disposition

No RefineGEMM CUDA kernel was implemented because the predeclared oracle gate
failed. This is a deliberate result, not an environment block.

The fixed-contract design remains technically expressible: a persistent CTA
pool could issue token-major tiles for high-M_e experts and weight/NK-parallel
tiles for tiny experts within one owner-local launch. It would preserve packed
row identity, expert weights, route weights and output order. However, the data
show only one narrow alternative-kernel region (`M_e=1`), existing grouped GEMM
already wins every mixed real invocation, and an actual two-subgroup realization
loses badly. A mega-kernel cannot justify its complexity against a 1.43--1.66%
credible dense E2E ceiling.

Reopening the design would require a new measured BF16 kernel with a materially
broader tiny/medium winning region and an updated one-launch oracle above 8%.
Changing only the production vLLM tuning should be treated as backend tuning,
not RefineGEMM.


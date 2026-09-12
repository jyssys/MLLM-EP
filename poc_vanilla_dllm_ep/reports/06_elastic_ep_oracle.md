# Elastic EP-degree oracle

EP1/EP2/EP4 are replayed on identical hidden inputs and token counts with the
same fused expert primitive. The operator oracle chooses the fastest degree per
shape with zero transition cost; the request oracle reweights those shapes by
the captured vanilla trace and must then charge any resident-replica,
workspace, KV/layout, and synchronization transition cost.


The identical-input layer replay covers M=1,2,4,8,16,32,64,128,256 with five
warm-ups and 30 measured samples per point. EP1 wins **every** shape. Its median
MoE latency rises from 1.178 ms (M=1) to 1.591 ms (M=256); EP2 spans
1.604–1.686 ms and EP4 1.684–1.652 ms. There is no small/large-work crossover
in the natural diffusion range.

The request-weighted oracle independently joins the same logical MoE calls
across topology traces, groups them into whole denoising/prefill steps, and
chooses one EP degree per step. EP1 wins every step. Best static EP1 and the
zero-transition dynamic oracle are identical, giving **0% direct E2E
headroom** before charging weight residency, KV/layout, graph, or switching
cost.

**Decision: KILL.** Using EP1 continuously may be appropriate for isolated
latency when the 30B model barely fits, but that is a capacity/topology choice,
not an elastic-EP research method; it leaves essentially no HBM headroom for
serving concurrency.

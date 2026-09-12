# Prior-Art Boundary

This is intentionally a bounded audit because the current-runtime oracle is the
first gate. Continuous batching chooses which requests enter a ready set; RAWS
would choose how many already-ready requests form one physical EP wave. TBO and
DeepEP microbatching overlap or pipeline work; generic adaptive batching changes
batch granularity using queue or workload size. Epoch compacts decision-live
diffusion work, changing the physical rows on which a future wave-size policy
could operate.

Accordingly, merely selecting a different `mini_batch_size` is not a research
contribution. A viable distinction would require all of: refinement-dependent
sparse shape, topology-specific winner changes, material gain over the best
static setting, and a nontrivial policy after accounting for queue delay. The
current PoC does not perform a full literature/implementation survey unless its
measured oracle passes 5%.

Reference context is retained in the parent report
`poc_flashvep/reports/llada2_flash_100b_ep_scaling_method_poc.md`; no claim of
novelty is made here before the headroom gate.

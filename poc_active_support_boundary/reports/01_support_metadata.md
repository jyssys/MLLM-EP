# Active-support and metadata controls

We kept the **same 256 routed rows and BF16 expert arithmetic per row**,
changing the number of active owner-local experts. A separate uniform vs
high-variance experiment keeps both active count and total rows matched
while changing the `M_e` distribution. Each setting uses 3 independent
Python/GPU restarts, 8 warmups and 40 CUDA-event measurements per restart;
operation order is randomized. Full64 always has the complete resident
weights and zero-length expert offsets. The optimistic active-only variant
gets the active weights and new offsets **precomputed outside timing**,
without any cost of dynamic expert-ID lookup, pointer indirection or
weight-copy; real sparse IDs require a gather that is also excluded.

| 256 total rows; uniform | active 1 | 4 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| Full64 `torch._grouped_mm` expert MLP (ms) | 0.070 | 0.090 | 0.215 | 0.372 | 0.655 |
| Optimistic active-only (ms) | 0.096 | 0.092 | 0.213 | 0.371 | 0.653 |

**Support sensitivity is real, but does not identify inactive-descriptor
overhead.** More *active* experts require more distinct useful expert weight
work/tiles even with unchanged total expert row FLOPs; shortening the
zero-length descriptors offers virtually no benefit. At active16,
256 total rows, high-variance vs uniform Full64 was 0.218 vs 0.215 ms:
the variation control is much smaller than the active-support sweep.
Active-only even regresses by 36% at active1: this optimistic dynamic
descriptor variant is not a uniformly valid production improvement.
All [65 controlled geometries](../SYNTHETIC_CONTROLS.csv) are reported.

In the **entire** available real corpus (18 dense + 28 nonempty
future-compacted local route geometries) with 3 independent restarts,
the clipped optimistic active-only saving is 0.15% of expert stage on
either dense task (0.038% of projected request E2E); hypothetical
compacted 0.28--0.32% expert stage, 0.07--0.08% request upper bound.
See [REQUEST_ORACLES.csv](../REQUEST_ORACLES.csv).

## Actual DeepEP/vLLM handoff versus standalone offsets control

Current installed
`vllm/.../deepep_ht_prepare_finalize.py` calls DeepEP
`get_dispatch_layout` and `Buffer.dispatch` and receives a per-expert
Python `expert_num_tokens_per_expert_list`. Its `_receiver` passes that
list into `ExpertTokensMetadata.make_from_list(...,device=expert_x.device)`;
the source comments `Makes a GPU-CPU copy`. This is a concrete metadata
handoff, **not** proof that rebuilding `torch._grouped_mm` offsets is
currently part of the full-model path. For the operator replay,
prebuilt GPU offsets beat rebuilt GPU-cumsum offsets by about
14--16 µs per local stage; this is an **operator wrapper** contrast.

Standalone sync-including host-visible medians over three restarts:
`make_from_list` around 20 µs, GPU cumsum around 18--20 µs, and
CPU-list-to-CUDA-offset around 36--38 µs across active counts 1--64.
They do not show a full-model request-critical 20--38 µs that a new
kernel can entirely remove. Using both `make_from_list` and CPU offsets
as separate simultaneous production waste would double count two
different execution paths. Native prebuilt `torch._grouped_mm` **still**
runs `prepare_grouped_gemm_data` twice, about 2.3 µs GPU each in Nsight;
that kernel is not inactive-descriptor tax, and direct metadata
consumption would need to prove what portion it actually eliminates.
[METADATA_SUMMARY.csv](../METADATA_SUMMARY.csv) records the standalone
controls. Current upstream DeepEP V2 already exposes GPU per-expert
prefix metadata; claiming `GPU offsets exist` as an original method
would collide with that interface.

# Decision — active-support-aware fused expert MLP PoC

**NO-GO for a new CUDA kernel on this LLaDA2.0-Flash BF16 H100
true-EP4 substrate.** Active support affects latency substantially,
but the effect is caused by work on *actually active* expert weights:
optimistically discarding zero-length descriptors saves 0.15% of
expert stage on real dense routes, corresponding to only **~0.04%
projected request E2E**. MLP boundaries are real separate kernels,
but O4 combined credible additional E2E is only **0.80% GSM8K /
0.78% HumanEval**, compared with an assumed strongest
`torch._grouped_mm` EP4 replacement. Even erasing the directly
observed boundary altogether gives only 2.63% / 2.33% projected
E2E for dense routes. There is no 8% implementation gate; we wrote
no new CUDA kernel, integrated runtime or novel selector.

## Evidence and gates

| Control / oracle | What was actually done | Dense GSM8K | Dense HumanEval |
|---|---|---:|---:|
| 256 rows, active1→64 | 3-process owner-local BF16 grouped replay, fixed useful row FLOPs | 0.070→0.655 ms | same controlled geometry |
| O1 only inactive descriptor removal | measured optimistic prebuilt active-only replay; fused comparison preserved | **0.038% projected E2E** | **0.038%** |
| O2 impossible observed-component boundary-free | zero standalone activation/weight and repeated prep, no useful GEMM FLOPs removed | **2.63% upper bound** | **2.33%** |
| O2 + unmeasured-HBM generous memcpy | intentional double counting; not a causal work saving | **3.76% sensitivity** | **3.35%** |
| O3 credible boundary target | 60% of Nsight 3 elementwise + 2 prep GPU kernel-time envelope | **0.76% projection** | **0.74%** |
| O4 combined credible | O1 + O3 once, no offset/copy double count | **0.80% projection** | **0.78%** |

The synthetic active-support phenomenon is genuine, but the descriptor
mechanism **fails causal subtraction** (Full64 vs optimistically
active-only). High-variance/uniform controls at the same 256 rows
and active16 differ ~1.2% in grouped latency. In the whole available
real route corpus there are 18 dense and 28 future-known
live-compacted nonzero invocations; all were benchmarked across
3 independent restarts, 8 warmups/40 paired repetitions, randomized
operator order. Production vLLM fused remains in the comparison,
but its currently installed E64/N1024/H100 untuned default is weaker
than grouped: **we do not compare a new candidate only to weak vLLM
default.** Our `torch._grouped_mm` is owner-local tested, not already
integrated into the full LLaDA2 EP4 path.

Frozen same-day clean BCT for the strongest mini32 EP4 runtime
was 6.075 s GSM8K (NFE66, score 5/32) / 7.343 s HumanEval (NFE86,
score 6/32) over 3 prior independent restarts. Routed expert
critical-rank timing attributed ~29.5% of the clean BCT in the prior
observer-heavy profile. The previous production→grouped operator
gain of 22.08%/20.52% reduces the *hypothetical* post-grouped pie
to 24.56%/25.00% by Amdahl normalization. New projected E2E
figures multiply real local replay stage saving by that corrected
pie. **None is a newly measured clean integrated request speedup.**
They may overestimate realizable gains; they are useful kill gates.
No new final-sequence/GSM8K/HumanEval-quality experiment was
performed because there is no kernel/runtime intervention to test.
Numerical operator replay vs installed vLLM fused achieved
min cosine 0.9999957/max rel-L2 0.316% using synthetic BF16
weights and real per-expert row distributions.

Nsight Systems saw 26 GEMM, 26 `prepare_grouped_gemm_data`
and 39 elementwise kernel calls over 13 complete traced forward
passes: per MLP, two useful GEMMs, two ~2.3 µs descriptor-prep
kernels and three elementwise kernels. `torch._grouped_mm` does
run small repeat-prep despite offsets being supplied; this does not
prove 64 inactive groups were the reason. No hardware HBM traffic
counters were available; intermediate round-trip **analytic**
12 KiB/routed row and standalone copy proxy are reported separately
from directly observed kernel durations.

## The twenty requested answers

1. The ~29.5% E2E expert pie is **reused** from the prior matched
   EP4 profile (29.47%/29.55% observer-attributed), not freshly
   established clean request-critical mass in this PoC.
2. Yes, at exactly 256 total rows, full grouped expert stage rises
   from 0.070 ms (active1) to 0.655 ms (active64).
3. No substantial **inactive descriptor** tax survives the direct
   optimistic active-only subtraction: ~0.15% real dense expert time;
   additional active weight work is not removable.
4. The installed DeepEP HT→vLLM path does convert the receive
   expert-count Python list via `make_from_list`. The replay's
   explicit offsets rebuild is **not** proven to be the existing
   production operator path; upstream DeepEP V2 already has GPU
   prefix metadata.
5. Prebuilt versus GPU-cumsum offsets save about 14--16 µs in the
   isolated **replay wrapper**. Standalone production
   `make_from_list` host-visible median ~20 µs cannot be added
   as proven request savings.
6. Gate/up standalone timing is ~58--60% of the measured grouped
   expert stage on real dense routes.
7. SwiGLU isolated timing is ~4--6% of dense stage; hypothetical
   live-compacted HumanEval late rises to 12.4% relative, with much
   lower absolute stage time.
8. Gate/up and activated HBM write/read total 12,288 B per
   routed row **analytically**; actual HBM traffic/fraction was
   not measurable with available profiler, so no measured latency
   percentage is asserted.
9. Down standalone timing is ~36--38% of real dense stage.
10. Route-weight isolated replay is ~3.5--5.4% of real dense
    stage, but real vLLM/DeepEP already applies top-k
    weights and reduction in finalize.
11. Perfect **directly observed** component-zero boundary O2
    projects 2.63%/2.33% dense E2E, 2.77%/3.55% hypothetical
    live-compacted; adding a generous unmeasured HBM memcpy
    proxy deliberately double counts and tops out at
    5.04% only on hypothetical live-compacted HumanEval.
12. Credible O3 boundary projects 0.76%/0.74% dense E2E.
13. O1 active-only projects 0.038%/0.038% dense E2E.
14. O4 combined credible projects 0.80%/0.78% dense E2E
    (future-compacted 0.71%/0.88% optimistic sensitivity).
15. No, credible combined E2E is **well below 8%**.
16. Late *relative* boundary percentage grows but
    descriptor shortening barely helps even when
    future-compacted active support falls to ~20/~13.
17. Early/middle carry more **absolute** expert time; late
    can have a larger boundary percentage but low request
    gain. No refinement phase reaches a kernel promotion gate.
18. [SonicMoE](https://arxiv.org/html/2512.14080v2) already
    fuses SwiGLU with up-GEMM epilogue on BF16 Hopper, while
    down still consumes activated intermediate. Our local
    CUDA12.8/Torch2.8 misses its official CUDA12.9+/Torch2.11+
    prerequisite, so Sonic was source audited, not measured.
19. DeepEP legacy metadata handoff is concrete, but
    [DeepEP V2](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffers/elastic.py),
    [CUTLASS offsets](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/005_grouped_gemm_contiguous_offset.html)
    and SonicMoE already address much of the alleged new interface.
    A differentiation would require a **measured**
    request-critical improvement beyond those, absent here.
20. No paper-level custom CUDA-kernel headroom was
    established. The worthwhile follow-up is ordinary
    engineering: tune/integrate available grouped/fused
    expert implementations, only revisit a bespoke
    DeepEP-native expert path if a different model/batch
    shows >=8% credible request-level gap against a
    tuned strongest existing baseline.

The comparison and caveats live in [runtime/source audit](00_environment_runtime.md),
[support and metadata](01_support_metadata.md),
[boundary trace](02_boundary_nsys.md), [real oracles](03_replay_oracles.md),
[prior-art attack](04_prior_art.md) and
[ATTEMPT_LOG.csv](../ATTEMPT_LOG.csv). The actual
user-requested burn is resumed **only after GPU measurement ended**.

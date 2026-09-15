# Real refinement, correctness and request upper bounds

The previous true EP4 model trace supplies per-local-expert `M_e`
for each real route. Full corpus: 18 observed dense-local invocations,
plus 28 **nonempty** future-known live-compacted cases. Each used
three independent process/GPU restarts, 8 warmups, 40 CUDA-event
repetitions and randomized operator order. Empty compacted cases
were omitted because they execute **no expert kernel**; one early
metadata/zero-row probe exited 136 and was fixed by the explicit skip.
The BF16 packed owner-local operator replay matches vLLM fused
numerically with minimum cosine 0.9999957 and maximum relative L2
0.316%; numerical equivalence is **not** a complete trajectory
replacement test. The actual model's weights were **not** loaded by
the replay (shape/random BF16 weights), and no new integrated
request output was generated. Baseline bounded GSM8K/HumanEval quality
is reused from the frozen clean run, and the PoC changes no model
generation code.

O1 sets **only** inactive expert descriptor/offset/scheduler overhead
to zero by a faster preprepared active-only backend; expert arithmetic
and distinct weights remain. O2 is an *impossible perfect*
activation+route-weight+repeated-prep *observed-component* boundary
ceiling, counting
standalone activation runtime as if useful activation arithmetic could
be free. An additional `O2_plus_all_materialization_unrealistic`
explicitly adds a separate full-intermediate memcpy proxy **that can
double count observed activation** to bracket unmeasured gate/up HBM
materialization; dense request ceilings 3.76%/3.35%, future-compacted
3.92%/5.04%, still below the 8% kernel gate. O3 captures 60%
of the Nsight elementwise/prep envelope,
capped by O2 and full stage. O4 combines O1+O3, with no GPU offsets,
host sync or intermediate memcpy separately counted a second time.

| Task and scope | O1 expert / request | O2 perfect expert / request | O3 credible expert / request | O4 credible expert / request |
|---|---:|---:|---:|---:|
| GSM8K dense real routes | 0.15% / **0.04%** | 10.70% / **2.63%** | 3.09% / **0.76%** | 3.24% / **0.80%** |
| HumanEval dense real routes | 0.15% / **0.04%** | 9.30% / **2.33%** | 2.95% / **0.74%** | 3.11% / **0.78%** |
| GSM8K future-known compacted | 0.32% / **0.08%** | 11.29% / **2.77%** | 2.58% / **0.63%** | 2.90% / **0.71%** |
| HumanEval future-known compacted | 0.28% / **0.07%** | 14.20% / **3.55%** | 3.23% / **0.81%** | 3.51% / **0.88%** |

All request numbers are **projected optimistic Amdahl upper bounds,
not measured TTFT/BCT gain**. The previous observer-attributed
production-expert BCT pie 29.47%/29.55% is corrected for the already
stronger grouped operator: using the frozen *operator-replay*
22.08%/20.52% fused→grouped stage reduction gives hypothetical
grouped expert pie 24.56%/25.00% of an otherwise identical request.
Multiply stage-weighted saving by that post-grouped pie. This assumes
perfect integration, no new layout costs or CPU/GPU overlap and that
the sampled local-route corpus represents whole-request critical
expert work. Therefore the projectable figures could overestimate
realizable clean EP4 gain. Future-compacted figures are sensitivity
with the unchanged **pre-Epoch** BCT pie used only as a generous
upper bound; they must never be called actual post-Epoch speedup.
Full derivation and source data:
[REQUEST_ORACLES.csv](../REQUEST_ORACLES.csv),
[REAL_REFINEMENT_MAPPING.csv](../REAL_REFINEMENT_MAPPING.csv),
and [analysis script](../scripts/analyze_support_boundary.py).

In dense real routes, active support does **not** systematically
collapse (middle has 48/53 mean active experts on GSM/HE; early 46/51;
late 37/40). After *hypothetical* live compaction late falls to ~20/13
active experts, but that tiny regime has less absolute GPU expert
time, so credible request gain is still below 1%. Late benefits are
not multiplied by whole-request pie.

Promotion gates: credible additional request E2E <5% means **NO-GO**.
Both actual-dense and compacted O4 are <1%; even impossible O2 stays
below 3.6% with directly measured boundary components (the intentionally
double-counting materialization envelope reaches 5.04% only for
hypothetical post-compaction HumanEval, not measured request gain).
No new CUDA kernel
or full-model replacement is justified.

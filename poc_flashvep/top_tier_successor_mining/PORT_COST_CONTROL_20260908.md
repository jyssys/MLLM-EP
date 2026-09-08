# Port-cost falsification controls

The first clean ChartQA B1 and GQA B4 runs show higher natural request latency
for all non-stock reference ports. This is **not yet a method failure**. A
faithful decision rule can still have a needlessly costly implementation.

Ranked alternatives and tests:

1. Repeated per-layer token metadata work contributes appreciably. Cache only
   immutable token identities, TP padding slices and modality masks within one
   model forward; reset every forward. Compare all generated tokens to the
   uncached policy, and pair request latency. No routing equation changes.
2. Changed generated answer lengths explain the natural E2E difference. Force
   exactly32 generated tokens in a separate diagnostic. Its answers do not enter
   task-quality Pareto claims. Compare B1 and B16 offered cohort caps.
3. Nominal offered batch does not equal the scheduled decode population, because
   image rendering and natural EOS change membership. Log actual scheduled
   request identities and counts in a separate instrumented B16 run.
4. MoDES pays its masking cost on decode despite little useful expert-work
   reduction there. Count actual dropped assignments by phase, and test the
   obvious prefill-only fallback. This intentionally changes the method's
   application scope and is labeled **BASELINE + TRIVIAL FIX**, not original
   MoDES or a successor contribution.
5. EP-specific communication and tiny decode grouped-GEMM work outweigh any
   reduced expert working set. Inspect correctly joined rank-local stage events
   on a separate diagnostic; stage maxima are not summed and are not substituted
   for clean request-level measurements.

All controls use physical GPUs4–7 after the clean six-condition screen releases
them. They do not run concurrently with it. Original six-condition policies have
neither metadata caching nor a phase restriction, so these additions do not
change the reference treatment between those runs. The branch is bounded; no
custom kernel or general scheduler is introduced.

Source refinement (13:38 KST): official `MoDES/models/qwen3.py:483–505` constructs
the modality masks **once per model forward**, then assigns them in the layer
loop at530–535. Our per-layer reconstruction is a **port-lifetime overhead**, not
a newly discovered weakness in MoDES. The within-forward cached-mask variant
restores this reference lifetime. Verify output equivalence and rerun clean
MoDES serving on the corrected port before making final efficiency statements.

If a simple implementation fix recovers most of the apparent loss, report that
fact. If the port remains slow, do not extrapolate that all original-paper
hardware/runtime configurations would be slow. Native Libra's scale-mismatch
and its captured-VL rather than full-request boundary remain separate caveats.

## Completed metadata control

32 ChartQA requests ×3 repetitions,11 policies,one persistent engine,48 paired
two-DP cohort units per cached/reference contrast. All96 generated sequences per
cached policy equal its uncached reference exactly.

| Policy | Request E2E reduction from metadata reuse | Paired-cohort95% CI |
|---|---:|---:|
| MoDES70 | 9.6468% | [8.5908,10.6390]% |
| MoDES85 | 9.5523% | [7.9586,10.3350]% |
| SERE S2/.5 | .2642% | [-1.0479,.9064]% |
| SERE S4/.5 | 1.3329% | [.2547,2.0712]% |
| SERE S2/.7 | 1.3911% | [.4772,2.2030]% |

This is **our port correction**, not MoDES successor headroom. Corrected MoDES70
and85 are still5.50% and5.95% slower than matched vanilla on this32-request subset.
Do not extrapolate its quality loss to the128-question primary screen; its sampled
question difficulty differs. Full corrected-port serving rerun is required.

Independent metadata-only GPU diagnostic:4 contexts ×6 M values ×30 randomized
repetitions per implementation (1,440 observations).48 `isin` calls cost median
9.112ms at M1 and11.34–11.37ms at M4/16, versus1.65ms at M128–2048. A semantically
equivalent broadcast-membership control costs about .97–1.20ms per48 calls.
Profiler shows two `aten::_unique` calls and three `cudaStreamSynchronize` calls
for the M1 `isin` observation, absent in the broadcast control. The common
profiler-teardown `cudaDeviceSynchronize` is **not attributed to `isin`**. These
standalone event spans include host enqueue gaps, and are not additive serving
speedup estimates. The live cache contrast above establishes request-level cost.

Data: `analysis/resume_controls_20260908` and
`analysis/metadata_operator_control_20260908` under the result root.
# Follow-up: algorithmic no-op and output-lifetime control

`ep_port_control_decision_20260908`:32 ChartQA inputs,offered B16,forced32
output tokens,three paired cohort repetitions. SERE S8 (retain every original
top-k assignment) costs15.67% E2E versus vanilla; MoDES tau=0 costs7.82%.
Actual SERE S2/S4 costs17.51%/17.03%; MoDES70 costs7.70% in this control.
The overhead floor therefore exists without the intended rerouting/skipping
benefit. It cannot be described as a newly discovered MLLM method failure.

All96 paired outputs match vanilla through the first EOS and at the first token.
Full32 continuations match81/96 for SERE and87/96 for MoDES. Vanilla itself
matches only30/32 and29/32 across its repetitions after forcing generation past
EOS. Therefore do not claim full-trajectory bitwise identity. This is an
algorithmic decision-identity cost control; the official primitive tests are the
separate exact-routing checks. Three cohort units do not establish cross-engine
confidence. Details:`analysis/noop_cost_parity_20260908/summary.json`.

Prefill-only MoDES,64 ChartQA inputs/B1/three reps, keeps the observed quality
score of each all-phase policy unchanged, but only reduces its penalty from
about5.9% to2.9% versus vanilla. It remains slower than vanilla in this control.
This is an obvious phase-gating attack,not a proposed successor method.

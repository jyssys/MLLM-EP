# Policy implementation

Status: `DENSE_SENSITIVITY_AND_PHYSICAL_OPERATOR_PROBE_IMPLEMENTED`.
Model and runtime identities appear in `00_runtime_audit.md`.

## Implemented scope

| Component | File | Status |
|---|---|---|
| Frozen manifests and partitions | `refinement_ep_allocation/prepare.py` | Implemented and executed |
| Ownership-safe burner control | `refinement_ep_allocation/resources.py` | Implemented and executed |
| Official HF generation worker | `refinement_ep_allocation/hf_worker.py` | User-approved 128/32 pilot complete |
| Physical EP4 worker | `refinement_ep_allocation/ep4_worker.py` | User-approved 128/32 pilot complete |
| Identity policy hook | `refinement_ep_allocation/ep4_worker.py` | 32 HumanEval requests and both group smoke tests passed |
| Original full baseline queue | `refinement_ep_allocation/campaign.py` | Canceled by user scope amendment |
| Durable pilot queue | `refinement_ep_allocation/pilot.py` | Complete |
| Causal confidence observer | `refinement_ep_allocation/refinement_observer.py` | 16 retained-all validation trajectories complete |
| Dispatch trace analysis | `refinement_ep_allocation/trace_analysis.py` | Executed on fresh sanity traces |
| Full-trajectory edge intervention | `refinement_ep_allocation/semantic_emulator.py` | Dense diagnostic, not acceleration |
| Durable sensitivity queue | `refinement_ep_allocation/sensitivity_campaign.py` | Independent EP4 groups |
| Separate validation scoring | `refinement_ep_allocation/sensitivity_analysis.py` | Frozen parser, labels hidden from online worker |
| P2–P5 structural replay | `refinement_ep_allocation/allocation_replay.py` | TRACE_ESTIMATE |
| Native operator microbenchmark | `refinement_ep_allocation/operator_probe.py` | Actual Mini activations and BF16 weights |
| Isolated A/B cost queue | `refinement_ep_allocation/cost_campaign.py` | Other EP4 group remains quiet |

Paths in this table are relative to `poc_flashvep/`.
The existing dInfer and model source files remain unchanged.
The workers reuse the existing physical DeepEP bridge.

## Approximate policy ladder

| Policy | Proposed information | Physical sparse implementation |
|---|---|---|
| P1 uniform | Independent seeded random edge scores | Dense quality diagnostic only |
| P2 router-only | Current selected router weights | Dense quality diagnostic and physical operator probe |
| P3 refinement-only | Causal refinement state and current router | Dense quality diagnostic only |
| P4 EP-only | Current router and current destination load | Fixed-trajectory replay only |
| P5 joint | Causal refinement state, current router, destination load | Fixed-trajectory replay only |
| P6 oracle | Future outcomes in offline analysis only | NOT_RUN |

Sparse runtime engineering remains gated by held-out sensitivity and realistic EP headroom.
The pilot does not replace full quality confirmation for a later positive claim.
No full-generation policy removes expert kernels yet.
The separate operator probe can remove assignments before dispatch.
Its timings do not establish an implemented request speedup.

## Dense intervention and exact budgets

The intervention retains all current router IDs and sets selected original weights to zero before dispatch.
All original expert kernels still execute.
Actual dispatch-edge reduction and expert-kernel reduction are therefore both zero.
Only the contribution to the generated result changes.
This is the Phase 2 diagnostic, not a Phase 4 implementation.

Each source rank selects `floor(local_rows × 8 × fraction)` contributions.
One protected slot per token preserves minimum k=1.
Its index is `(local_row + rank + layer + forward_index) mod 8`.
All diagnostic policies use the same protection and target budget.
Retained weights remain unchanged and unnormalized.

P1 uses an independent seeded Torch generator.
It does not change the decoder's random stream.
P2 scores edges by their current normalized router weight.
P3 multiplies that score by a causal token-risk factor.
Committed positions use risk 0.75.
Masked positions use `1 + sigmoid((c−0.70)/0.05) × sigmoid((0.95−c)/0.05)`.
Here `c` comes from a previously completed decode forward.
Missing history sets the frontier term to zero.
The fallback still uses the fresh current router and current mask state.

The diagnostic obtains confidence through an extra 157,184-entry softmax after the original decode decision.
That operation is not a free baseline feature.
P3 is `REFLEX_INSPIRED_PROXY`, not a faithful REFLEX implementation.
It does not reuse stale hidden states, routes, or expert outputs.
Earlier raw summaries use a constant REFLEX-proxy description even for P1/P2.
Their explicit `kind` field and launch command identify the actual method.
The current metadata distinguishes uniform, router-only, and refinement-proxy methods.
This metadata correction does not change arithmetic or old raw records.

P4/P5 replay divides P2/P3 scores by squared relative destination load.
It uses the current call's global rank counts.
An online implementation would need a count exchange or another causal estimate.
No replay row proves the cost of that exchange.

## Physical operator boundary

The probe captures actual native activations at physical MoE layers 1 and 19.
It retains original BF16 expert weights, shared experts, and source-row aggregation.
It tests global row counts 32, 64, and 128 when those shapes occur.
It replaces selected source expert IDs with −1 before DeepEP dispatch.
Receive-buffer counts verify removal of physical assignments.
The native sparse runner executes the remaining expert assignments.

The removal mask is precomputed from the current router during warm-up.
Online selection and confidence costs are excluded from this optimistic operator comparison.
This path is not an end-to-end sparse generation engine.

## Failures and regression checks

An initial Group A zero-removal run failed exact parity.
The revised path retains modified buffers until the existing request-end GPU synchronization completes.
The revised control passed all 16 A/B validation requests.
An independent baseline restart matched all eight repeated even-index requests.
These results support the revised path, but do not establish the unique cause of the initial failure.

A structured hash gave a biased P1 selection.
Its runs remain preserved as superseded diagnostics.
The final P1 uses an independent random generator.

Regression tests exposed copied flatten views for non-contiguous masks.
Explicit row/slot indexing fixes the Torch and NumPy cases.
Tests cover both layouts, exact budgets, minimum k, and RNG isolation.
New starts record weight strides and assert actual zero-mask counts.

Resource control retries only vanished process snapshots.
It retains every ownership assertion and does not signal unrelated jobs.

## Required tests before sparse execution

1. Preserve exact output and NFE with all edges retained.
2. Preserve current router selection before subset allocation.
3. Verify actual omitted dispatch entries and expert work.
4. Verify per-token minimum k and original retained weights.
5. Verify EOS, tail blocks, tiny counts, and zero-remote cases.
6. Verify rank metadata conservation and output order.
7. Measure policy overhead without heavy tracing.

Shared experts, attention, residuals, and model weights must remain unchanged.
Missing refinement history must use the documented fallback or retain full work.

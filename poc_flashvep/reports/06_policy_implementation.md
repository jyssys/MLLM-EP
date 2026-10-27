# Policy implementation

Status: `BASELINE_IMPLEMENTED_APPROXIMATE_POLICIES_NOT_RUN`.
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
| Causal confidence observer | `refinement_ep_allocation/refinement_observer.py` | Non-test trace collection in progress |
| Dispatch trace analysis | `refinement_ep_allocation/trace_analysis.py` | Executed on fresh sanity traces |

Paths in this table are relative to `poc_flashvep/`.
The existing dInfer and model source files remain unchanged.
The workers reuse the existing physical DeepEP bridge.

## Approximate policy ladder

| Policy | Proposed information | Physical sparse implementation |
|---|---|---|
| P1 uniform | Edge budget and deterministic positions | NOT_RUN |
| P2 router-only | Current selected router weights | NOT_RUN |
| P3 refinement-only | Causal refinement state and current router | NOT_RUN |
| P4 EP-only | Current router and calibrated EP cost | NOT_RUN |
| P5 joint | Causal refinement state, current router, calibrated EP cost | NOT_RUN |
| P6 oracle | Future outcomes in offline analysis only | NOT_RUN |

Sparse runtime engineering remains gated by held-out sensitivity and realistic EP headroom.
The pilot does not replace full quality confirmation for a later positive claim.
No policy removes expert kernels yet.
No result claims an implemented sparse speedup.
Masking an output after full expert computation would not meet the implementation contract.

## Required tests before sparse execution

1. Preserve exact output and NFE with all edges retained.
2. Preserve current router selection before subset allocation.
3. Verify actual omitted dispatch entries and expert work.
4. Verify per-token minimum k and original retained weights.
5. Verify EOS, tail blocks, tiny counts, and zero-remote cases.
6. Verify rank metadata conservation and output order.
7. Measure policy overhead without heavy tracing.

Shared experts, attention, residuals, and model weights must remain unchanged.
Missing refinement history must use a documented full-work fallback.

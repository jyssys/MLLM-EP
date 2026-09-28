# Q2-3 Visual-Grounding Budget Gate — frozen bounded specification

## Question

At a current decode site `(request, step t, layer l)`, can a request-specific
visual-prefill representation determine whether reducing the entire native
Top-8 mixture to native-router-tail K6/K4 is functionally safe?

This changes the unit from the failed Q2/Q2-2 individual expert path to the
whole token/layer budget. It does not identify which expert should later be
removed. EP-aware active-expert collapse remains a separately gated Phase 2.

## Fixed baseline and scope

- Qwen3-VL-30B-A3B-Instruct revision
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16.
- TRUE TP2-DP2-EP4-PP1 with DeepEP on physical GPUs 2,3,6,7.
- Universal-prompt detailed-caption cohort, because request-specific text is
  held byte-identical and request information resides in the image.
- Same-domain image swap with the same teacher-forced original continuation.
- Instrumented diagnostic runs only: no TPOT/E2E/speedup claim.
- No MoDES implementation or comparison in this stage.

## Frozen unit and cohort

- 8 calibration requests and 24 image-hash-disjoint fresh held-out requests.
- All Q2 and Q2-2 request/image identities are excluded.
- One predeclared layer/step site per request, balanced over layers
  `[4,12,24,36,44,47]` and steps `[16,32,48,64,96,112]`.
- Original and same-domain counterfactual visual-token counts differ by at
  most 10%.
- Selection never uses routing, grounding, or damage results.

## Single-pass grounding signal

At the target layer, capture at most 16 visual and 16 non-visual prefill router
input rows per TP shard. Construct uncentered rank-4 right-singular subspaces
from row-normalized samples. For the current decode pre-MoE hidden `h`:

`E_m = ||U_m normalize(h)||_2^2`, for `m in {vision,text,full}`.

The frozen primary score is:

`S_sep = log((E_vision + 1e-6) / (E_text + 1e-6))`.

Controls are `E_text`, `E_full`, and the native router removed-mass score.
No feature is selected after held-out damage is inspected.

## Structured functional damage

At the exact captured hidden row, retain the native highest-weight K6 or K4
paths, zero the remaining original paths, and renormalize the retained weights
to sum to one. No replacement expert is selected. The diagnostic kernel may
still execute zero-weight rows, so this is semantic evidence, not physical
work reduction.

For each K in {6,4}:

`D_K = max(0, target_logprob_drop) + 2*JSD + 1[top1_flip]`.

Primary structured damage is fixed as `D_struct = (D_6 + D_4)/2` at the
current step. A four-token teacher-forced horizon is secondary.

## Held-out gate

All confidence intervals use a 10,000-repeat request-cluster bootstrap.
Promotion requires all of:

1. `Spearman(S_sep_orig, D_struct_orig) > 0` and its bootstrap 95% CI lower
   bound is above zero.
2. `Spearman(S_sep_orig-S_sep_cf, D_struct_orig-D_struct_cf) > 0` and its
   bootstrap 95% CI lower bound is above zero.
3. `S_sep` exceeds the strongest frozen text/full/router control in held-out
   Spearman, with the paired bootstrap increment CI lower bound above zero.

Failure stops before expert selection, modified-route runtime, or latency
measurement. A merely positive point estimate is unresolved, not a pass.

## Verdicts

- `VISUAL_GROUNDING_BUDGET_SIGNAL_SUPPORTED`
- `VISUAL_GROUNDING_ASSOCIATION_ONLY`
- `GENERIC_UNIT_RISK_ONLY`
- `NO_STRUCTURED_DAMAGE_SIGNAL`
- `RUNTIME_BLOCKED`

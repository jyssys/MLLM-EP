# Phase 2 Stub Interface Report

Date: 2026-06-24

## Scope

This report summarizes the Phase 2 integration stubs prepared after Phase 1.
The work only fixes interfaces for future GPU/model integration. No real
Qwen3-VL forward execution, hook registration, interleaved M-RoPE inversion,
DeepSpeed all-to-all dispatch, expert forward, speed measurement, or accuracy
measurement was implemented.

Every new Phase 2 entry point is limited to:

- function signature
- type hints
- docstring describing input/output tensors
- `NotImplementedError("TODO(Phase2): ...")`

## Stubs Added

### Forward Hook Registration

File: `hooks/register_hooks.py`

Interfaces:

- `CapturedLayerTensors`
- `ExpertAssignment`
- `HookHandles`
- `register_calibration_hooks(model, hook_plan, ...) -> HookHandles`
- `extract_routing(captured, *, top_k, num_experts, normalize_topk_prob=True) -> ExpertAssignment`
- `build_calibration_payload(captured, routing) -> dict[str, torch.Tensor]`
- `remove_hooks(handles) -> None`

The interface reflects the hook points documented in `docs/model_arch.md`:

- model/generate entry for `input_ids`, `mm_token_type_ids`, and vision/text masks
- decoder self-attention modules for per-layer attention weights
- `Qwen3VLMoeTextSparseMoeBlock` for router logits and expert-input hidden states

The output tensor contract matches Phase 1 calibration:

- `expert_assignment`: `[num_tokens, top_k]`
- `routing_weights`: `[num_tokens, top_k]`
- `router_logits`: `[num_tokens, num_experts]`
- `token_type`: `[num_tokens]`
- `hidden_states`: `[num_tokens, hidden_dim]`
- `attention`: `[batch, heads, seq, seq]` or a compatible collapsed attention tensor

### de-RoPE / M-RoPE Importance

File: `method2/derope.py`

Interface:

- `derope_attention(q, k, cos, sin, position_ids_3d, *, mrope_section=(24, 20, 20), mrope_interleaved=True, attention_mask=None) -> torch.Tensor`

`method2/importance.py` now routes `compute_cross_attention_importance(..., derope=True, ...)`
to this stub. The raw Phase 1 path remains unchanged.

The signature explicitly captures Qwen3-VL M-RoPE requirements:

- `position_ids_3d`: `[4, batch, seq]`
- row 0: text/cache position
- rows 1-3: temporal, height, width ids
- default `mrope_section=(24, 20, 20)`
- default `mrope_interleaved=True`

The docstring records the important boundary: a 1D RoPE inverse is incorrect
for image/video tokens because Qwen3-VL uses 3D interleaved M-RoPE.

### DeepSpeed EP Dispatch Insertion

File: `pipeline/ep_integration.py`

Interfaces:

- `MergeFn`
- `CapFn`
- `ep_moe_forward_with_merge(hidden, router_out, placement, merge_fn, cap_fn, *, importance, ...) -> torch.Tensor`
- `restore_ep_moe_forward(model, *, patched_modules) -> None`

The wrapper contract fixes the future Method 2 insertion point:

1. read router top-k assignments
2. identify straggler GPU load
3. apply cap logic so the straggler stops at second-place load
4. call Phase 1 `expert_aware_merge`
5. insert merged representations after routing and before EP dispatch
6. run future DeepSpeed all-to-all dispatch, expert forward, and combine

The `merge_fn` and `cap_fn` protocol shapes are compatible with:

- `method2.merge.expert_aware_merge`
- `method2.cap.cap_merge_count`

## Existing Modules Touched

File: `method2/importance.py`

`compute_cross_attention_importance` gained optional Phase 2-only parameters:

- `q`
- `k`
- `cos`
- `sin`
- `position_ids_3d`
- `mrope_section`
- `mrope_interleaved`

These are only used when `derope=True`, which currently raises the Phase 2
TODO error through `derope_attention`.

File: `docs/model_arch.md`

The Phase 2 TODO boundary section now lists the prepared stub files and states
that they are signature/docstring/`NotImplementedError` only.

## Interface Tests

File: `tests/test_phase2_stubs.py`

The tests verify:

- stub functions exist with the expected signature
- hook capture dataclasses match Phase 1 calibration tensor shapes
- `collect_layer_stats` accepts the future hook payload shapes
- `compute_cross_attention_importance(..., derope=True, ...)` reaches the de-RoPE stub
- EP integration accepts `expert_aware_merge` and `cap_merge_count` as callable interfaces
- every Phase 2 stub raises the expected `NotImplementedError`

Verification:

```text
python3 -m pytest -q
26 passed in 1.32s
```

## Preserved Boundaries

Still not implemented:

- real HF/Qwen3-VL hook registration
- real model forward
- de-RoPE math
- CLS importance term
- redundant-token rerouting
- DeepSpeed EP dispatch/combine
- GPU kernels
- speedup and accuracy measurement
- hyperparameter tuning


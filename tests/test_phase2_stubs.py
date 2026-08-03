import inspect

import pytest
import torch

from calib.collect_stats import TOKEN_TEXT, TOKEN_VISION, build_mode_hook_plan, collect_layer_stats
from hooks.register_hooks import (
    CapturedLayerTensors,
    ExpertAssignment,
    HookHandles,
    build_calibration_payload,
    extract_routing,
    register_calibration_hooks,
    remove_hooks,
)
from method1.placement import PlacementResult
from method2.cap import cap_merge_count
from method2.derope import derope_attention
from method2.importance import compute_cross_attention_importance
from method2.merge import expert_aware_merge
from pipeline.ep_integration import ep_moe_forward_with_merge, restore_ep_moe_forward


def _captured_layer() -> CapturedLayerTensors:
    return CapturedLayerTensors(
        layer_idx=0,
        router_logits=torch.zeros(4, 8),
        attention=torch.zeros(1, 2, 4, 4),
        hidden_states=torch.zeros(4, 16),
        token_type=torch.tensor([TOKEN_TEXT, TOKEN_VISION, TOKEN_VISION, TOKEN_TEXT]),
        mm_token_type_ids=torch.tensor([[0, 1, 1, 0]]),
        input_ids=torch.tensor([[101, 151655, 151655, 102]]),
    )


def _assignment() -> ExpertAssignment:
    return ExpertAssignment(
        expert_assignment=torch.tensor([[0, 1], [2, 3], [2, 4], [0, 5]]),
        routing_weights=torch.full((4, 2), 0.5),
        router_logits=torch.zeros(4, 8),
    )


def test_hook_registration_stub_signature_and_phase2_errors():
    params = inspect.signature(register_calibration_hooks).parameters
    assert list(params)[:2] == ["model", "hook_plan"]
    assert "capture_router_logits" in params
    assert "capture_attentions" in params
    assert "capture_hidden_states" in params
    assert "capture_token_masks" in params

    plan = build_mode_hook_plan()
    with pytest.raises(NotImplementedError, match="TODO\\(Phase2\\): register against real Qwen3-VL forward"):
        register_calibration_hooks(object(), plan)

    handles = HookHandles(handles=(), captured_by_layer={})
    with pytest.raises(NotImplementedError, match="TODO\\(Phase2\\): remove real Qwen3-VL hook handles"):
        remove_hooks(handles)


def test_hook_capture_types_match_calibration_payload_shapes():
    captured = _captured_layer()
    routing = _assignment()

    assert captured.router_logits.shape == (4, 8)
    assert captured.attention.shape == (1, 2, 4, 4)
    assert captured.hidden_states.shape == (4, 16)
    assert captured.token_type.shape == (4,)
    assert routing.expert_assignment.shape == (4, 2)
    assert routing.routing_weights.shape == (4, 2)

    stats = collect_layer_stats(
        expert_assignment=routing.expert_assignment,
        token_type=captured.token_type,
        hidden_states=captured.hidden_states,
        num_experts=8,
    )
    assert stats["centroid"].shape == (8, 16)

    with pytest.raises(NotImplementedError, match="TODO\\(Phase2\\): extract Qwen3-VL router top-k assignments"):
        extract_routing(captured, top_k=2, num_experts=8)
    with pytest.raises(NotImplementedError, match="TODO\\(Phase2\\): materialize calibration payload"):
        build_calibration_payload(captured, routing)


def test_derope_stub_signature_and_importance_connection():
    params = inspect.signature(derope_attention).parameters
    assert list(params)[:5] == ["q", "k", "cos", "sin", "position_ids_3d"]
    assert "mrope_section" in params
    assert "mrope_interleaved" in params

    q = torch.zeros(1, 2, 4, 64)
    k = torch.zeros(1, 2, 4, 64)
    cos = torch.ones(1, 1, 4, 64)
    sin = torch.zeros(1, 1, 4, 64)
    position_ids_3d = torch.zeros(4, 1, 4, dtype=torch.long)

    with pytest.raises(NotImplementedError, match="TODO\\(Phase2\\): invert interleaved M-RoPE on Q/K"):
        derope_attention(q, k, cos, sin, position_ids_3d, mrope_section=(24, 20, 20), mrope_interleaved=True)

    with pytest.raises(NotImplementedError, match="TODO\\(Phase2\\): invert interleaved M-RoPE on Q/K"):
        compute_cross_attention_importance(
            torch.zeros(4, 4),
            [0],
            [1],
            derope=True,
            q=q,
            k=k,
            cos=cos,
            sin=sin,
            position_ids_3d=position_ids_3d,
            mrope_section=(24, 20, 20),
            mrope_interleaved=True,
        )


def test_ep_integration_stub_accepts_phase1_merge_and_cap_interfaces():
    params = inspect.signature(ep_moe_forward_with_merge).parameters
    assert list(params)[:5] == ["hidden", "router_out", "placement", "merge_fn", "cap_fn"]
    assert "importance" in params
    assert "candidate_mask_or_indices" in params
    assert "gpu_loads" in params

    router_out = _assignment()
    placement = PlacementResult(
        expert_to_gpu=torch.tensor([0, 0, 1, 1, 0, 1, 0, 1]),
        gpu_loads=torch.tensor([4.0, 4.0]),
        expert_counts=torch.tensor([4, 4]),
    )

    with pytest.raises(NotImplementedError, match="TODO\\(Phase2\\): DeepSpeed all-to-all dispatch"):
        ep_moe_forward_with_merge(
            hidden=torch.zeros(4, 16),
            router_out=router_out,
            placement=placement,
            merge_fn=expert_aware_merge,
            cap_fn=cap_merge_count,
            importance=torch.ones(4),
            candidate_mask_or_indices=torch.tensor([1, 2]),
            gpu_loads=torch.tensor([6.0, 3.0]),
            straggler_gpus=torch.tensor([0]),
        )

    with pytest.raises(NotImplementedError, match="TODO\\(Phase2\\): restore patched"):
        restore_ep_moe_forward(object(), patched_modules={})

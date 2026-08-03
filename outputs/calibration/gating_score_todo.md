# P2 Gating Score Status

Status: TODO(Phase2B or deeper vLLM hook).

vLLM 0.20 exposes `CompletionOutput.routed_experts` when
`enable_return_routed_experts=True`, but that path stores only top-k expert ids.
The local source path is:

- `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`
- `RoutedExpertsCapturer.capture(layer_id, topk_ids)`
- output field: `CompletionOutput.routed_experts`

`topk_weights`/router softmax values exist inside the fused MoE routing path, but
are not exported by the public routed-expert capture API. This phase does not
patch vLLM fused kernels or alter the model, so P2 gating heatmaps are left as
TODO while P1/P3 use actual vLLM EP routed expert ids.

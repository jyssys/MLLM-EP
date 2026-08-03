# Qwen3-VL-30B-A3B-Instruct Architecture Notes

Checked on 2026-06-24 against:

- Hugging Face model repo: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
- Local config: `models/Qwen3-VL-30B-A3B-Instruct/config.json`
- Transformers source reference: https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
- Transformers docs: https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl_moe
- MODE reference implementation: https://github.com/MingZwhy/MODE/blob/main/mllm_quant/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
- MODE frequency recorder: https://github.com/MingZwhy/MODE/blob/main/mllm_quant/moe_freq/record_freq.py

## High-Level Model

- Architecture class: `Qwen3VLMoeForConditionalGeneration`
- HF `model_type`: `qwen3_vl_moe`
- Text decoder layers: 48
- Experts per MoE layer: 128
- Experts selected per token: 8 (`num_experts_per_tok`)
- Decoder hidden size: 2048
- Attention heads: 32
- KV heads: 4
- Head dim: 128
- Dense MLP intermediate size: 6144
- MoE expert intermediate size: 768
- Sparse schedule: `decoder_sparse_step=1`, `mlp_only_layers=[]`; therefore every decoder layer uses the sparse MoE block in the shipped config.
- Router top-k probability normalization: enabled in config (`norm_topk_prob=true`) and implemented by dividing selected top-k probabilities by their selected sum.

Vision side:

- Vision depth: 27
- Vision hidden size: 1152
- Vision output hidden size: 2048
- Patch size: 16
- Spatial merge size: 2
- Temporal patch size: 2
- DeepStack visual feature layers: 8, 16, 24

## Router, Dispatch, Expert, Combine Path

The text decoder layer chooses `Qwen3VLMoeTextSparseMoeBlock` whenever the layer is not in `mlp_only_layers` and `(layer_idx + 1) % decoder_sparse_step == 0`. With this model config, that means all 48 text decoder layers are sparse MoE layers.

Relevant source locations in Transformers v5.12.0:

- `Qwen3VLMoeTextTopKRouter`: source lines 128-144
- `Qwen3VLMoeTextSparseMoeBlock`: source lines 147-158
- `Qwen3VLMoeTextExperts`: source lines 90-125
- `Qwen3VLMoeTextDecoderLayer`: source lines 320-363

Forward path:

1. `Qwen3VLMoeTextSparseMoeBlock.forward` flattens hidden states from `[batch, seq, hidden]` to `[tokens, hidden]`.
2. Router computes logits with a bias-free linear projection: `router_logits = F.linear(hidden_states, self.weight)`, where `weight` has shape `[num_experts, hidden_size]`.
3. Router applies softmax over experts, takes top-k, normalizes the selected top-k probabilities, and returns `router_logits`, `routing_weights`, and `selected_experts`.
4. `Qwen3VLMoeTextExperts.forward` builds a one-hot expert mask from selected top-k experts, iterates over experts that received tokens, gathers `(top_k_pos, token_idx)` with `torch.where`, and runs each expert FFN on the gathered token states.
5. Each expert is a SwiGLU-style FFN stored in packed tensors:
   - `gate_up_proj`: `[num_experts, 2 * moe_intermediate_size, hidden_size]`
   - `down_proj`: `[num_experts, hidden_size, moe_intermediate_size]`
6. The expert output is multiplied by the selected router weight for that token/top-k position.
7. Combine is `final_hidden_states.index_add_(0, token_idx, weighted_expert_output)`.

Phase 1 implication: the dummy code should treat routing as plain `expert_assignment` plus optional `routing_weights`. DeepSpeed EP all-to-all dispatch is not present in this HF eager path and is a Phase 2 integration point.

MODE's `qwen3_vl_moe` adapter exposes the same Phase 2 hook points: the sparse MoE block has a bias-free `gate`, computes `router_logits = gate(hidden_states)`, applies softmax and top-k, normalizes selected weights, dispatches to experts, and returns `(routed_out, router_logits)`. MODE's recorder first tries to recover routing from returned router logits and otherwise recomputes `gate(hidden_states)`. Our Phase 1 calibration code mirrors that expected output as plain `expert_assignment` tensors; it does not import or execute MODE/HF model code.

## RoPE / M-RoPE

This model uses multimodal RoPE, not a plain 1D text-only RoPE.

Config:

- `rope_theta`: 5000000
- `rope_scaling.rope_type`: `default`
- `rope_scaling.mrope_interleaved`: `true`
- `rope_scaling.mrope_section`: `[24, 20, 20]`

Relevant source locations:

- `Qwen3VLMoeTextRotaryEmbedding`: lines 735-823
- Text model position handling: lines 886-912
- 3D multimodal position generation: lines 1051-1142
- Attention applies rotary to query/key before attention: lines 265-293

How it is applied:

- The conditional-generation wrapper computes 3D multimodal position ids when multimodal inputs and `mm_token_type_ids` are provided.
- `mm_token_type_ids` distinguish text/image/video as `0/1/2`.
- For each modality run, text positions are ordinary increasing positions expanded to three rows. Image/video positions are generated as temporal, height, width indices from `grid_thw`.
- The language model may receive `position_ids` with shape `[4, batch, seq]`: row 0 is the text-position id used for causal mask/cache handling, and rows 1-3 are temporal/height/width ids used by rotary embedding.
- `Qwen3VLMoeTextRotaryEmbedding.forward` expands ordinary 2D position ids to 3D when needed, computes three frequency tensors, then calls `apply_interleaved_mrope`.
- `apply_interleaved_mrope` starts from the temporal frequency tensor and overwrites interleaved positions with height and width frequencies according to `mrope_section`. The intended layout is interleaved T/H/W frequency allocation rather than simple chunked T-then-H-then-W.
- Decoder self-attention then applies standard rotary rotation to query and key using the resulting cos/sin tensors.

Phase 2 de-RoPE implication: a future de-RoPE implementation must invert Q/K rotary using the same 3D position ids and interleaved section layout. A 1D RoPE inverse would be wrong for image/video tokens.

MODE's vendored Qwen3-VL-MoE code also uses `mrope_section` and `apply_interleaved_mrope`, so it agrees with the Transformers reading: this is M-RoPE with interleaved temporal/height/width rotary allocation, not ordinary 1D RoPE.

## Vision vs Text Token Identification

Special token ids from config:

- `vision_start_token_id`: 151652
- `vision_end_token_id`: 151653
- `image_token_id`: 151655
- `video_token_id`: 151656

Runtime path:

- `get_placeholder_mask` detects image/video placeholder positions from `input_ids == image_token_id` and `input_ids == video_token_id`.
- Vision encoder outputs are inserted into the language-model embedding sequence with `masked_scatter` at those placeholder positions.
- `visual_pos_masks` marks the same visual positions and is used by DeepStack to add intermediate visual features into early decoder hidden states.
- `mm_token_type_ids` are required for correct M-RoPE when multimodal grids are present: text is 0, image is 1, video is 2.

Phase 1 dummy code should therefore model modality with explicit masks or token-type tensors. It should not depend on Qwen tokenizer objects or real pixel inputs.

## Phase 2 TODO Boundaries

- `# TODO(Phase2)`: real Qwen3-VL forward hooks for router logits, attentions, hidden states, and token type ids.
- `# TODO(Phase2)`: de-RoPE importance using the M-RoPE details above.
- `# TODO(Phase2)`: CLS attention term and lambda tuning.
- `# TODO(Phase2)`: DeepSpeed EP integration and all-to-all dispatch insertion point.
- `# TODO(Phase2)`: redundant token rerouting using collected centroids.
- `# TODO(Phase2)`: speedup, throughput, accuracy, and hyperparameter sweeps.

Interface stubs prepared for these boundaries:

- `hooks/register_hooks.py`: `register_calibration_hooks`, `extract_routing`, `build_calibration_payload`, `remove_hooks`
- `method2/derope.py`: `derope_attention`
- `pipeline/ep_integration.py`: `ep_moe_forward_with_merge`, `restore_ep_moe_forward`

All of these stubs are signature/docstring/`NotImplementedError` only. They are
intended to pin the Phase 2 interfaces without running real Qwen3-VL forward,
interleaved M-RoPE inversion, DeepSpeed all-to-all dispatch, or GPU logic.

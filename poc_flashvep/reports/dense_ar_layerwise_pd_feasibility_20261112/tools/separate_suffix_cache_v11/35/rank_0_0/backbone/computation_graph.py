from __future__ import annotations
import torch
class GraphModule(torch.nn.Module):
    def forward(self, s26: "Sym(s26)", L_residual_: "bf16[s26, 4096]", L_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_: "bf16[4096]", s59: "Sym(s26)", L_hidden_: "bf16[s26, 4096]", L_self_modules_layers_modules_35_modules_self_attn_modules_qkv_proj_parameters_weight_: "bf16[6144, 4096]", L_self_modules_layers_modules_35_modules_self_attn_modules_q_norm_parameters_weight_: "bf16[128]", L_self_modules_layers_modules_35_modules_self_attn_modules_k_norm_parameters_weight_: "bf16[128]", L_self_modules_layers_modules_35_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_: "bf16[40960, 128]", L_positions_: "i64[s26]", SYNTHETIC_LOCAL_tmp_0_ : vllm_utils_torch_utils_LayerName, L_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_: "bf16[4096, 4096]", L_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_35_modules_mlp_modules_gate_up_proj_parameters_weight_: "bf16[24576, 4096]", L_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_: "bf16[4096, 12288]", L_self_modules_norm_parameters_weight_: "bf16[4096]"):
        l_residual_ = L_residual_
        l_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_ = L_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_
        l_hidden_ = L_hidden_
        l_self_modules_layers_modules_35_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_35_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_layers_modules_35_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_layers_modules_35_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_layers_modules_35_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_layers_modules_35_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_ = L_self_modules_layers_modules_35_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_
        l_positions_ = L_positions_
        synthetic_local_tmp_0_ = SYNTHETIC_LOCAL_tmp_0_
        l_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_35_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_

        # No stacktrace found for following nodes
        submod_0 = self.submod_0(l_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_, l_hidden_, s26, l_residual_, l_self_modules_layers_modules_35_modules_self_attn_modules_qkv_proj_parameters_weight_, l_self_modules_layers_modules_35_modules_self_attn_modules_q_norm_parameters_weight_, l_self_modules_layers_modules_35_modules_self_attn_modules_k_norm_parameters_weight_, l_positions_, l_self_modules_layers_modules_35_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_);  l_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_ = l_hidden_ = l_residual_ = l_self_modules_layers_modules_35_modules_self_attn_modules_qkv_proj_parameters_weight_ = l_self_modules_layers_modules_35_modules_self_attn_modules_q_norm_parameters_weight_ = l_self_modules_layers_modules_35_modules_self_attn_modules_k_norm_parameters_weight_ = l_positions_ = l_self_modules_layers_modules_35_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_ = None
        getitem = submod_0[0]
        getitem_1 = submod_0[1]
        getitem_2 = submod_0[2]
        getitem_3 = submod_0[3]
        getitem_4 = submod_0[4]
        getitem_5 = submod_0[5];  submod_0 = None
        submod_1 = self.submod_1(getitem, s26, getitem_1, synthetic_local_tmp_0_, getitem_2, getitem_3);  getitem = getitem_1 = synthetic_local_tmp_0_ = getitem_2 = submod_1 = None
        submod_2 = self.submod_2(getitem_3, s26, l_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_, getitem_4, l_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_, getitem_5, l_self_modules_layers_modules_35_modules_mlp_modules_gate_up_proj_parameters_weight_, l_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_, l_self_modules_norm_parameters_weight_);  getitem_3 = s26 = l_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_ = getitem_4 = l_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_ = getitem_5 = l_self_modules_layers_modules_35_modules_mlp_modules_gate_up_proj_parameters_weight_ = l_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_ = l_self_modules_norm_parameters_weight_ = None
        return (submod_2,)

    class submod_0(torch.nn.Module):
        def forward(self, l_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_: "bf16[4096]", l_hidden_: "bf16[s26, 4096]", s26: "Sym(s26)", l_residual_: "bf16[s26, 4096]", l_self_modules_layers_modules_35_modules_self_attn_modules_qkv_proj_parameters_weight_: "bf16[6144, 4096]", l_self_modules_layers_modules_35_modules_self_attn_modules_q_norm_parameters_weight_: "bf16[128]", l_self_modules_layers_modules_35_modules_self_attn_modules_k_norm_parameters_weight_: "bf16[128]", l_positions_: "i64[s26]", l_self_modules_layers_modules_35_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_: "bf16[40960, 128]"):
            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/layernorm.py:91 in forward_native, code: self.weight.data if self.pass_weight_add else None,
            _get_data_attr: "bf16[4096]" = torch._C._autograd._get_data_attr(l_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_);  l_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_ = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/ir/op.py:534 in __call__, code: return self.torch_op(*args, **kwargs)
            fused_add_rms_norm_maybe_inplace = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(l_hidden_, l_residual_, _get_data_attr, 1e-06, None);  l_hidden_ = l_residual_ = _get_data_attr = None
            getitem: "bf16[s26, 4096]" = fused_add_rms_norm_maybe_inplace[0]
            getitem_1: "bf16[s26, 4096]" = fused_add_rms_norm_maybe_inplace[1];  fused_add_rms_norm_maybe_inplace = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
            t: "bf16[4096, 6144]" = l_self_modules_layers_modules_35_modules_self_attn_modules_qkv_proj_parameters_weight_.t();  l_self_modules_layers_modules_35_modules_self_attn_modules_qkv_proj_parameters_weight_ = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:146 in matmul_persistent, code: c = torch.empty((M, N), device=a.device, dtype=dtype)
            empty: "bf16[s26, 6144]" = torch.empty((s26, 6144), device = device(type='cuda', index=0), dtype = torch.bfloat16)

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/triton/__init__.py:68 in cdiv, code: return (x + y - 1) // y
            add: "Sym(s26 + 128)" = s26 + 128
            sub: "Sym(s26 + 127)" = add - 1;  add = None
            floordiv: "Sym(((s26 + 127)//128))" = sub // 128;  sub = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:153 in grid, code: triton.cdiv(M, META["BLOCK_SIZE_M"])
            mul: "Sym(48*(((s26 + 127)//128)))" = floordiv * 48

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:151 in grid, code: min(
            sym_min: "Sym(Min(132, 48*(((s26 + 127)//128))))" = torch.sym_min(132, mul);  mul = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:184 in matmul_persistent, code: matmul_kernel_persistent[grid](
            triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 145, constant_args_idx = 144, grid = [(sym_min, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': getitem, 'b_ptr': t, 'c_ptr': empty, 'M': s26});  sym_min = getitem = t = triton_kernel_wrapper_mutation = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/qwen3.py:159 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            split = empty.split([4096, 1024, 1024], dim = -1);  empty = None
            getitem_2: "bf16[s26, 4096]" = split[0]

            # No stacktrace found for following nodes
            sym_size_int: "Sym(s26)" = torch.ops.aten.sym_size.int(getitem_2, 0)

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/qwen3.py:159 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            getitem_3: "bf16[s26, 1024]" = split[1]

            # No stacktrace found for following nodes
            sym_size_int_1: "Sym(s26)" = torch.ops.aten.sym_size.int(getitem_3, 0)

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/qwen3.py:159 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            getitem_4: "bf16[s26, 1024]" = split[2];  split = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/qwen3.py:161 in forward, code: q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            view: "bf16[s26, 32, 128]" = getitem_2.view(s26, 32, 128);  getitem_2 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/layernorm.py:83 in forward_native, code: self.weight.data if self.pass_weight else None,
            _get_data_attr_1: "bf16[128]" = torch._C._autograd._get_data_attr(l_self_modules_layers_modules_35_modules_self_attn_modules_q_norm_parameters_weight_);  l_self_modules_layers_modules_35_modules_self_attn_modules_q_norm_parameters_weight_ = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/ir/op.py:372 in __call__, code: return self.torch_op(*args, **kwargs)
            rms_norm_default: "bf16[s26, 32, 128]" = torch.ops.vllm_ir.rms_norm.default(view, _get_data_attr_1, 1e-06, None);  view = _get_data_attr_1 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/qwen3.py:163 in forward, code: q = q_by_head.view(q.shape)
            view_1: "bf16[s26, 4096]" = rms_norm_default.view(sym_size_int, 4096);  rms_norm_default = sym_size_int = None

            # No stacktrace found for following nodes
            sym_size_int_2: "Sym(s26)" = torch.ops.aten.sym_size.int(view_1, 0)

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/qwen3.py:164 in forward, code: k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            view_2: "bf16[s26, 8, 128]" = getitem_3.view(s26, 8, 128);  getitem_3 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/layernorm.py:83 in forward_native, code: self.weight.data if self.pass_weight else None,
            _get_data_attr_2: "bf16[128]" = torch._C._autograd._get_data_attr(l_self_modules_layers_modules_35_modules_self_attn_modules_k_norm_parameters_weight_);  l_self_modules_layers_modules_35_modules_self_attn_modules_k_norm_parameters_weight_ = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/ir/op.py:372 in __call__, code: return self.torch_op(*args, **kwargs)
            rms_norm_default_1: "bf16[s26, 8, 128]" = torch.ops.vllm_ir.rms_norm.default(view_2, _get_data_attr_2, 1e-06, None);  view_2 = _get_data_attr_2 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/qwen3.py:166 in forward, code: k = k_by_head.view(k.shape)
            view_3: "bf16[s26, 1024]" = rms_norm_default_1.view(sym_size_int_1, 1024);  rms_norm_default_1 = sym_size_int_1 = None

            # No stacktrace found for following nodes
            sym_size_int_3: "Sym(s26)" = torch.ops.aten.sym_size.int(view_3, 0)

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:171 in forward_static, code: positions = positions.flatten()
            flatten: "i64[s26]" = l_positions_.flatten();  l_positions_ = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:173 in forward_static, code: cos_sin = cos_sin_cache.index_select(0, positions)
            index_select: "bf16[s26, 128]" = l_self_modules_layers_modules_35_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_.index_select(0, flatten);  l_self_modules_layers_modules_35_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_ = flatten = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:174 in forward_static, code: cos, sin = cos_sin.chunk(2, dim=-1)
            chunk = index_select.chunk(2, dim = -1);  index_select = None
            getitem_5: "bf16[s26, 64]" = chunk[0]
            getitem_6: "bf16[s26, 64]" = chunk[1];  chunk = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:177 in forward_static, code: query = query.view(num_tokens, -1, head_size)
            view_4: "bf16[s26, 32, 128]" = view_1.view(s26, -1, 128);  view_1 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:178 in forward_static, code: query_rot = query[..., :rotary_dim]
            getitem_7: "bf16[s26, 32, 128]" = view_4[(Ellipsis, slice(None, 128, None))]

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:179 in forward_static, code: query_pass = query[..., rotary_dim:]
            getitem_8: "bf16[s26, 32, 0]" = view_4[(Ellipsis, slice(128, None, None))];  view_4 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:166 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
            unsqueeze: "bf16[s26, 1, 64]" = getitem_5.unsqueeze(-2)
            to: "bf16[s26, 1, 64]" = unsqueeze.to(torch.bfloat16);  unsqueeze = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
            unsqueeze_1: "bf16[s26, 1, 64]" = getitem_6.unsqueeze(-2)
            to_1: "bf16[s26, 1, 64]" = unsqueeze_1.to(torch.bfloat16);  unsqueeze_1 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:170 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
            chunk_1 = torch.chunk(getitem_7, 2, dim = -1);  getitem_7 = None
            getitem_9: "bf16[s26, 32, 64]" = chunk_1[0]
            getitem_10: "bf16[s26, 32, 64]" = chunk_1[1];  chunk_1 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:175 in forward_static, code: o1 = x1 * cos - x2 * sin
            mul_1: "bf16[s26, 32, 64]" = getitem_9 * to
            mul_2: "bf16[s26, 32, 64]" = getitem_10 * to_1
            sub_1: "bf16[s26, 32, 64]" = mul_1 - mul_2;  mul_1 = mul_2 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:176 in forward_static, code: o2 = x2 * cos + x1 * sin
            mul_3: "bf16[s26, 32, 64]" = getitem_10 * to;  getitem_10 = to = None
            mul_4: "bf16[s26, 32, 64]" = getitem_9 * to_1;  getitem_9 = to_1 = None
            add_1: "bf16[s26, 32, 64]" = mul_3 + mul_4;  mul_3 = mul_4 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:179 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
            cat: "bf16[s26, 32, 128]" = torch.cat((sub_1, add_1), dim = -1);  sub_1 = add_1 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:186 in forward_static, code: query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
            cat_1: "bf16[s26, 32, 128]" = torch.cat((cat, getitem_8), dim = -1);  cat = getitem_8 = None
            reshape: "bf16[s26, 4096]" = cat_1.reshape(sym_size_int_2, 4096);  cat_1 = sym_size_int_2 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:191 in forward_static, code: key = key.view(num_tokens, -1, head_size)
            view_5: "bf16[s26, 8, 128]" = view_3.view(s26, -1, 128);  view_3 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:192 in forward_static, code: key_rot = key[..., :rotary_dim]
            getitem_11: "bf16[s26, 8, 128]" = view_5[(Ellipsis, slice(None, 128, None))]

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:193 in forward_static, code: key_pass = key[..., rotary_dim:]
            getitem_12: "bf16[s26, 8, 0]" = view_5[(Ellipsis, slice(128, None, None))];  view_5 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:166 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
            unsqueeze_2: "bf16[s26, 1, 64]" = getitem_5.unsqueeze(-2);  getitem_5 = None
            to_2: "bf16[s26, 1, 64]" = unsqueeze_2.to(torch.bfloat16);  unsqueeze_2 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
            unsqueeze_3: "bf16[s26, 1, 64]" = getitem_6.unsqueeze(-2);  getitem_6 = None
            to_3: "bf16[s26, 1, 64]" = unsqueeze_3.to(torch.bfloat16);  unsqueeze_3 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:170 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
            chunk_2 = torch.chunk(getitem_11, 2, dim = -1);  getitem_11 = None
            getitem_13: "bf16[s26, 8, 64]" = chunk_2[0]
            getitem_14: "bf16[s26, 8, 64]" = chunk_2[1];  chunk_2 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:175 in forward_static, code: o1 = x1 * cos - x2 * sin
            mul_5: "bf16[s26, 8, 64]" = getitem_13 * to_2
            mul_6: "bf16[s26, 8, 64]" = getitem_14 * to_3
            sub_2: "bf16[s26, 8, 64]" = mul_5 - mul_6;  mul_5 = mul_6 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:176 in forward_static, code: o2 = x2 * cos + x1 * sin
            mul_7: "bf16[s26, 8, 64]" = getitem_14 * to_2;  getitem_14 = to_2 = None
            mul_8: "bf16[s26, 8, 64]" = getitem_13 * to_3;  getitem_13 = to_3 = None
            add_2: "bf16[s26, 8, 64]" = mul_7 + mul_8;  mul_7 = mul_8 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/common.py:179 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
            cat_2: "bf16[s26, 8, 128]" = torch.cat((sub_2, add_2), dim = -1);  sub_2 = add_2 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/rotary_embedding/base.py:200 in forward_static, code: key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
            cat_3: "bf16[s26, 8, 128]" = torch.cat((cat_2, getitem_12), dim = -1);  cat_2 = getitem_12 = None
            reshape_1: "bf16[s26, 1024]" = cat_3.reshape(sym_size_int_3, 1024);  cat_3 = sym_size_int_3 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:531 in forward, code: output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
            size = torch.Size([s26, 4096]);  s26 = None
            empty_1: "bf16[s26, 4096]" = torch.empty(size, dtype = torch.bfloat16, device = device(type='cuda', index=0));  size = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:536 in forward, code: query = query.view(-1, self.num_heads, self.head_size)
            view_6: "bf16[s26, 32, 128]" = reshape.view(-1, 32, 128);  reshape = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:537 in forward, code: output = output.view(-1, self.num_heads, self.head_size_v)
            view_7: "bf16[s26, 32, 128]" = empty_1.view(-1, 32, 128);  empty_1 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:539 in forward, code: key = key.view(-1, self.num_kv_heads, self.head_size)
            view_8: "bf16[s26, 8, 128]" = reshape_1.view(-1, 8, 128);  reshape_1 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:541 in forward, code: value = value.view(-1, self.num_kv_heads, self.head_size_v)
            view_9: "bf16[s26, 8, 128]" = getitem_4.view(-1, 8, 128);  getitem_4 = None
            return (view_8, view_9, view_6, view_7, floordiv, getitem_1)

    class submod_1(torch.nn.Module):
        def forward(self, key_2: "bf16[s26, 8, 128]", s26: "Sym(s26)", value: "bf16[s26, 8, 128]", synthetic_local_tmp_0_ : vllm_utils_torch_utils_LayerName, query_2: "bf16[s26, 32, 128]", output_3: "bf16[s26, 32, 128]"):
            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:571 in forward, code: kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
            unified_kv_cache_update: "bf16[0]" = torch.ops.vllm.unified_kv_cache_update(key_2, value, synthetic_local_tmp_0_)

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:574 in forward, code: torch.ops.vllm.unified_attention_with_output(
            unified_attention_with_output = torch.ops.vllm.unified_attention_with_output(query_2, key_2, value, output_3, synthetic_local_tmp_0_, kv_cache_dummy_dep = unified_kv_cache_update);  query_2 = key_2 = value = output_3 = synthetic_local_tmp_0_ = unified_kv_cache_update = unified_attention_with_output = None
            return ()

    class submod_2(torch.nn.Module):
        def forward(self, output_3: "bf16[s26, 32, 128]", s26: "Sym(s26)", l_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_: "bf16[4096, 4096]", floordiv: "Sym(((s26 + 127)//128))", l_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_: "bf16[4096]", residual: "bf16[s26, 4096]", l_self_modules_layers_modules_35_modules_mlp_modules_gate_up_proj_parameters_weight_: "bf16[24576, 4096]", l_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_: "bf16[4096, 12288]", l_self_modules_norm_parameters_weight_: "bf16[4096]"):
            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:582 in forward, code: return output.view(-1, hidden_size)
            view: "bf16[s26, 4096]" = output_3.view(-1, 4096);  output_3 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
            t: "bf16[4096, 4096]" = l_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_.t();  l_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_ = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:146 in matmul_persistent, code: c = torch.empty((M, N), device=a.device, dtype=dtype)
            empty: "bf16[s26, 4096]" = torch.empty((s26, 4096), device = device(type='cuda', index=0), dtype = torch.bfloat16)

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:153 in grid, code: triton.cdiv(M, META["BLOCK_SIZE_M"])
            mul: "Sym(32*(((s26 + 127)//128)))" = floordiv * 32

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:151 in grid, code: min(
            sym_min: "Sym(Min(132, 32*(((s26 + 127)//128))))" = torch.sym_min(132, mul);  mul = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:184 in matmul_persistent, code: matmul_kernel_persistent[grid](
            triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 146, constant_args_idx = 145, grid = [(sym_min, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': view, 'b_ptr': t, 'c_ptr': empty, 'M': s26});  view = t = triton_kernel_wrapper_mutation = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/layernorm.py:91 in forward_native, code: self.weight.data if self.pass_weight_add else None,
            _get_data_attr: "bf16[4096]" = torch._C._autograd._get_data_attr(l_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_);  l_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_ = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/ir/op.py:534 in __call__, code: return self.torch_op(*args, **kwargs)
            fused_add_rms_norm_maybe_inplace = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(empty, residual, _get_data_attr, 1e-06, None);  empty = residual = _get_data_attr = None
            getitem: "bf16[s26, 4096]" = fused_add_rms_norm_maybe_inplace[0]
            getitem_1: "bf16[s26, 4096]" = fused_add_rms_norm_maybe_inplace[1];  fused_add_rms_norm_maybe_inplace = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
            t_1: "bf16[4096, 24576]" = l_self_modules_layers_modules_35_modules_mlp_modules_gate_up_proj_parameters_weight_.t();  l_self_modules_layers_modules_35_modules_mlp_modules_gate_up_proj_parameters_weight_ = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:146 in matmul_persistent, code: c = torch.empty((M, N), device=a.device, dtype=dtype)
            empty_1: "bf16[s26, 24576]" = torch.empty((s26, 24576), device = device(type='cuda', index=0), dtype = torch.bfloat16)

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:153 in grid, code: triton.cdiv(M, META["BLOCK_SIZE_M"])
            mul_1: "Sym(192*(((s26 + 127)//128)))" = floordiv * 192;  floordiv = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:151 in grid, code: min(
            sym_min_1: "Sym(Min(132, 192*(((s26 + 127)//128))))" = torch.sym_min(132, mul_1);  mul_1 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:184 in matmul_persistent, code: matmul_kernel_persistent[grid](
            triton_kernel_wrapper_mutation_1 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 147, constant_args_idx = 146, grid = [(sym_min_1, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': getitem, 'b_ptr': t_1, 'c_ptr': empty_1, 'M': s26});  sym_min_1 = getitem = t_1 = triton_kernel_wrapper_mutation_1 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/activation.py:143 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
            getitem_2: "bf16[s26, 12288]" = empty_1[(Ellipsis, slice(None, 12288, None))]
            silu: "bf16[s26, 12288]" = torch.nn.functional.silu(getitem_2);  getitem_2 = None
            getitem_3: "bf16[s26, 12288]" = empty_1[(Ellipsis, slice(12288, None, None))];  empty_1 = None
            mul_2: "bf16[s26, 12288]" = silu * getitem_3;  silu = getitem_3 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
            t_2: "bf16[12288, 4096]" = l_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_.t();  l_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_ = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:146 in matmul_persistent, code: c = torch.empty((M, N), device=a.device, dtype=dtype)
            empty_2: "bf16[s26, 4096]" = torch.empty((s26, 4096), device = device(type='cuda', index=0), dtype = torch.bfloat16)

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/batch_invariant.py:184 in matmul_persistent, code: matmul_kernel_persistent[grid](
            triton_kernel_wrapper_mutation_2 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 148, constant_args_idx = 147, grid = [(sym_min, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_2, 'b_ptr': t_2, 'c_ptr': empty_2, 'M': s26});  sym_min = mul_2 = t_2 = s26 = triton_kernel_wrapper_mutation_2 = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/layernorm.py:91 in forward_native, code: self.weight.data if self.pass_weight_add else None,
            _get_data_attr_1: "bf16[4096]" = torch._C._autograd._get_data_attr(l_self_modules_norm_parameters_weight_);  l_self_modules_norm_parameters_weight_ = None

            # File: /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/ir/op.py:534 in __call__, code: return self.torch_op(*args, **kwargs)
            fused_add_rms_norm_maybe_inplace_1 = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(empty_2, getitem_1, _get_data_attr_1, 1e-06, None);  empty_2 = getitem_1 = _get_data_attr_1 = None
            getitem_4: "bf16[s26, 4096]" = fused_add_rms_norm_maybe_inplace_1[0];  fused_add_rms_norm_maybe_inplace_1 = None
            return getitem_4

"""Correctness-only tensor witnesses from the actual standard TP worker."""
import os
from pathlib import Path
from ..duetserve_agg_decode.tp_worker import ExactTPWorker


class TensorWitnessWorker(ExactTPWorker):
    def witness_enable(self):
        import torch
        from .common import ROOT
        import json
        cfg = json.loads(Path(os.environ['DENSE_CONFIG']).read_text())
        self.witness = {}
        self.witness_handles = []
        self.witness_step = -1
        self.witness_full_shapes = {}
        self.witness_row_name = None
        self.witness_layer = None
        self.witness_write_counts = {}
        model = self.model_runner.get_model()
        wanted = ('embed_tokens', 'input_layernorm', 'qkv_proj', 'q_norm', 'k_norm',
                  'rotary_emb', 'attn', 'o_proj', 'post_attention_layernorm',
                  'gate_up_proj', 'act_fn', 'down_proj', 'norm')

        def store(key, value):
            if isinstance(value, torch.Tensor):
                self.witness_write_counts[key] = self.witness_write_counts.get(key, 0) + 1
                self.witness_full_shapes[key] = list(value.shape)
                if value.ndim >= 1:
                    value = value[-8:]
                self.witness[key] = value.detach().clone().cpu()
            elif isinstance(value, (tuple, list)):
                for i, element in enumerate(value):
                    store(key + '.' + str(i), element)

        def begin(_module, _args, _kwargs):
            self.witness_step += 1
            positions = _kwargs.get('positions', _args[1] if len(_args) > 1 else None)
            store(str(self.witness_step) + ':positions', positions)

        # This runtime invokes the decorated inner model forward directly.
        # Its nn.Module pre-hook does not run. Wrap the outer model forward.
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        original_forward = Qwen3ForCausalLM.forward
        def traced_forward(module, *args, **kwargs):
            begin(module, args, kwargs)
            return original_forward(module, *args, **kwargs)
        Qwen3ForCausalLM.forward = traced_forward
        for name, module in model.named_modules():
            if type(module).__name__ == 'Qwen3DecoderLayer':
                def enter_layer(_module, _args, name=name):
                    self.witness_layer = int(name.split('.')[-1])
                self.witness_handles.append(module.register_forward_pre_hook(enter_layer))
            if name.split('.')[-1] not in wanted:
                continue
            def before(_module, args, kwargs, name=name):
                if name.endswith('rotary_emb'):
                    name = 'model.layers.' + str(self.witness_layer) + '.self_attn.rotary_emb'
                if name.endswith(('o_proj', 'down_proj')):
                    self.witness_row_name = name
                store(str(self.witness_step) + ':' + name + ':input', args)
                for key, value in kwargs.items():
                    store(str(self.witness_step) + ':' + name + ':input_' + key, value)
            def after(_module, _args, output, name=name):
                if name.endswith('rotary_emb'):
                    name = 'model.layers.' + str(self.witness_layer) + '.self_attn.rotary_emb'
                store(str(self.witness_step) + ':' + name + ':output', output)
                if name.endswith(('o_proj', 'down_proj')):
                    self.witness_row_name = None
            self.witness_handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self.witness_handles.append(module.register_forward_hook(after))
        from vllm.model_executor.layers import linear
        all_reduce = linear.tensor_model_parallel_all_reduce
        def reduce(value):
            name = self.witness_row_name or 'unknown_row_projection'
            key = str(self.witness_step) + ':' + name + ':collective'
            store(key + '_input', value)
            result = all_reduce(value)
            store(key + '_output', result)
            return result
        linear.tensor_model_parallel_all_reduce = reduce
        compute_logits = model.compute_logits
        def logits(hidden_states):
            store(str(self.witness_step) + ':lm_head:input', hidden_states)
            result = compute_logits(hidden_states)
            # The vocabulary axis must not be truncated.
            if result is not None:
                self.witness[str(self.witness_step) + ':logits'] = result.detach().clone().cpu()
            return result
        model.compute_logits = logits
        weights = {name: p.detach().clone().cpu() for name, p in model.named_parameters()
                   if name.startswith('model.layers.0.')}
        path = ROOT / 'raw' / cfg['name'] / ('weights_rank' + str(self.rank) + '.pt')
        torch.save(weights, path)
        return dict(rank=self.rank, model_type=str(type(model)), modules=len(self.witness_handles),
                    weight_tensors=len(weights), scope='CORRECTNESS_ONLY; tensor copies invalidate timing.')

    def witness_export(self):
        import json
        import torch
        from .common import ROOT, write
        cfg = json.loads(Path(os.environ['DENSE_CONFIG']).read_text())
        folder = ROOT / 'raw' / cfg['name']
        torch.save(self.witness, folder / ('tensors_rank' + str(self.rank) + '.pt'))
        write(folder / ('tensor_shapes_rank' + str(self.rank) + '.json'), self.witness_full_shapes)
        write(folder / ('tensor_write_counts_rank' + str(self.rank) + '.json'), self.witness_write_counts)
        assert self.witness_step + 1 == cfg['cap'], 'Missing outer-forward boundaries'
        assert max(self.witness_write_counts.values()) == 1, 'A shared module overwrote tensor provenance'
        return dict(rank=self.rank, calls=self.witness_step + 1, tensors=len(self.witness))

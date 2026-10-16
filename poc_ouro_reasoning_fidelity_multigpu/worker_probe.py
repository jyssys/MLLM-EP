"""Selected teacher-aligned states. Never use this observer for clean timing."""
import os
import re
from pathlib import Path
from poc_looped_recurrence_tp_4models.worker_probe import FourModelProbe

class FidelityProbe(FourModelProbe):
    def audit_effective_contract(self):
        import inspect
        import hashlib
        model=self.get_model()
        source=Path(inspect.getfile(type(model)))
        config=self.vllm_config.model_config
        return dict(physical_layers=len(model.model.layers),actual_recurrent_passes=model.model.total_ut_steps,
            logical_layer_executions_per_token=len(model.model.layers)*model.model.total_ut_steps,
            hf_eos_token_id=config.hf_config.eos_token_id,
            effective_eos_token_id='Read the driver input processor audit. Worker does not own the tokenizer.',
            precision=str(config.dtype),model_class=type(model).__name__,source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            topology=repr(self.vllm_config.parallel_config),cache_config=repr(self.vllm_config.cache_config))

    def fidelity_start(self, label, prompt_length, checkpoints):
        import numpy as np
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        self.fidelity_index = 0
        self.fidelity_records = []
        original = self.get_model().compute_logits
        self.fidelity_original = original
        rank = get_tensor_model_parallel_rank()
        def compute(hidden, *args, **kwargs):
            logits = original(hidden, *args, **kwargs)
            index = self.fidelity_index
            self.fidelity_index += 1
            if index in checkpoints:
                tables = self.model_runner.block_tables.input_block_tables
                assert len(tables) == 1
                table = tables[0][0].cpu().tolist()
                positions = sorted(set([0, prompt_length-1, prompt_length+index-1]))
                positions = [p for p in positions if p >= 0]
                slots = {}
                for name, module in self.vllm_config.compilation_config.static_forward_context.items():
                    cache = getattr(module, 'kv_cache', None)
                    if not isinstance(cache, torch.Tensor) or not cache.numel():
                        continue
                    slot = int(re.search(r'layers\.(\d+)\.', name).group(1))
                    assert cache.ndim == 4 and cache.shape[-1] == 256
                    coords = torch.tensor([0,1,2,3,124,125,126,127,128,129,130,131,252,253,254,255], device=cache.device)
                    slots[slot] = torch.stack([cache[table[p//16],:2,p%16,coords] for p in positions]).float().cpu().numpy()
                assert sorted(slots) == list(range(192))
                path = Path(os.environ['LOOPED_KV_OUTPUT']) / f'{label}.step{index}.rank{rank}.npz'
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path, logits=logits.detach().float().cpu().numpy(),
                    hidden=hidden.detach().float().cpu().numpy(),
                    KV=np.stack([slots[i] for i in range(192)]))
                self.fidelity_records.append(dict(step=index, path=str(path), rank=rank,
                    positions=positions, logical_slots=list(range(192)),
                    pre_norm_state='NOT_CAPTURED', logits_stage='native pre-sampler logits'))
            return logits
        self.get_model().compute_logits = compute
        return True

    def fidelity_finish(self):
        self.get_model().compute_logits = self.fidelity_original
        return dict(records=self.fidelity_records, actual_logits_calls=self.fidelity_index)

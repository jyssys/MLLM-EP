"""Native vLLM numerical observer. This is not a clean timing path."""
from poc_looped_recurrence_tp_4models.worker_probe import FourModelProbe


class ExternalProbe(FourModelProbe):
    def start_tp_reference(self):
        self.reference_rows=[];self.reference_original=self.get_model().compute_logits
        def compute(hidden,*args,**kwargs):
            logits=self.reference_original(hidden,*args,**kwargs)
            self.reference_rows.append(dict(logits=logits.detach().float().cpu(),hidden=hidden.detach().float().cpu()))
            return logits
        self.get_model().compute_logits=compute
        return True

    def finish_tp_reference(self,P,O,label):
        import re
        import torch
        from .contract import ROOT,write
        from vllm.distributed import get_tensor_model_parallel_rank
        self.get_model().compute_logits=self.reference_original
        rank=get_tensor_model_parallel_rank();table=self.model_runner.block_tables.input_block_tables[0][0].cpu().tolist()
        selected={};positions=[0,P-1,P+O-2]
        for name,module in self.vllm_config.compilation_config.static_forward_context.items():
            cache=getattr(module,'kv_cache',None)
            if not isinstance(cache,torch.Tensor) or not cache.numel():continue
            slot=int(re.search(r'layers\.(\d+)\.',name).group(1))
            assert cache.ndim==4 and cache.shape[-1]==256
            selected[slot]=torch.stack([cache[table[p//16],:,p%16,:] for p in positions]).cpu()
        path=ROOT/'.tensor_cache'/f'{label}_rank{rank}.pt'
        torch.save(dict(rows=self.reference_rows,KV=selected,positions=positions),path)
        return dict(rank=rank,path=str(path),rows=len(self.reference_rows),KV_slots=len(selected),positions=positions)

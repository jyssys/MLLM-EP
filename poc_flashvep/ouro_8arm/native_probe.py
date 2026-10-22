"""Native Ouro extension, isolated from all frozen worker extensions."""
from poc_looped_recurrence_tp_4models.worker_probe import FourModelProbe

class Probe(FourModelProbe):
    def profile_begin(self):
        import torch
        torch.cuda.cudart().cudaProfilerStart();torch.cuda.nvtx.range_push('EIGHT_COHORT')
        return True

    def profile_end(self):
        import torch
        torch.cuda.synchronize();torch.cuda.nvtx.range_pop();torch.cuda.cudart().cudaProfilerStop()
        return True

    def safety_start(self):
        import torch
        self.finite=torch.ones((),dtype=torch.bool,device=self.device)
        self.safety_calls=0
        self.safety_original=self.get_model().compute_logits
        def compute(*args,**kwargs):
            self.safety_calls+=1
            z=self.safety_original(*args,**kwargs);self.finite.logical_and_(torch.isfinite(z).all());return z
        self.get_model().compute_logits=compute
        return True
    def safety_finish(self):
        self.get_model().compute_logits=self.safety_original
        return bool(self.finite)
    def safety_audit(self):
        return dict(compute_logits_wrapper_calls=self.safety_calls,
            scope='Python wrapper calls in this cohort. A positive count checks that the installed observer was reached.')
    def start_trace(self):
        from vllm.distributed import get_tensor_model_parallel_rank
        from .numerical_trace import Trace
        self.trace=Trace(self.get_model(),native=True,rank=get_tensor_model_parallel_rank())
        self.logits=[];self.original_logits=self.get_model().compute_logits
        def logits(hidden,*args,**kwargs):
            z=self.original_logits(hidden,*args,**kwargs)
            if self.trace.rank==0:self.logits.append(z.detach().float().cpu())
            self.trace.step+=1
            return z
        self.get_model().compute_logits=logits
        return True
    def finish_trace(self,label,tp):
        import torch
        from .contract import ROOT,sha
        self.trace.close();self.get_model().compute_logits=self.original_logits
        if self.trace.rank:return dict(rank=self.trace.rank,steps=self.trace.step)
        path=ROOT/'.tensor_cache'/f'vLLM_TP{tp}_{label}.pt'
        torch.save(dict(frames=self.trace.frames,logits=self.logits),path)
        return dict(rank=0,path=str(path),sha256=sha(path),steps=self.trace.step,frames=len(self.trace.frames))

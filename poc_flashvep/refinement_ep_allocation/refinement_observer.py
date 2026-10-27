"""Causal confidence history for observer runs. Never reuse model states or expert outputs."""
import json
import numpy as np
import torch
from poc_dllm_ep_comm.instrumentation import Recorder
from .common import write

class RefinementRecorder(Recorder):
    def __init__(self,rank):
        super().__init__(rank)
        self.confidence=torch.full((8192,),float('nan'),device=f'cuda:{rank}',dtype=torch.float32)
        self.observed_at=torch.full((8192,),-1,device=f'cuda:{rank}',dtype=torch.int32)
        self.completed_decodes=0
        self.history_rows=[]

    def begin(self,mode,request_id,prompt_length):
        super().begin(mode,request_id,prompt_length)
        self.confidence.fill_(float('nan'))
        self.observed_at.fill_(-1)
        self.completed_decodes=0
        self.history_rows=[]

    def wrapped_stage(self,layer,stage,fn,wait=False):
        wrapped=super().wrapped_stage(layer,stage,fn,wait)
        def call(*args,**kwargs):
            history=None
            if self.mode=='trace' and stage=='dispatch':
                count=kwargs['topk_idx'].shape[0]
                start=self.rank*(self.context['positions'].numel()//4)
                positions=self.context['positions'].flatten()[start:start+count].long()
                history=dict(route_index=len(self.routes),call_id=self.call,layer=layer,
                    confidence=self.confidence.index_select(0,positions).clone(),
                    observed_at=self.observed_at.index_select(0,positions).clone(),
                    completed_decodes_before_dispatch=self.completed_decodes)
            result=wrapped(*args,**kwargs)
            if history is not None:
                self.history_rows.append(history)
            return result
        return call

    def install(self,model,dllm):
        super().install(model,dllm)
        decoder=dllm.decoder if hasattr(dllm,'decoder') else None
        assert decoder is not None,'Native decoder is unavailable for observer attachment'
        original=decoder.decode
        def decode(logits,block_start,block_end,x,*args,**kwargs):
            result=original(logits,block_start,block_end,x,*args,**kwargs)
            if self.mode=='trace':
                # This is an extra diagnostic softmax, not a claim about baseline decoder internals.
                # It executes only after the current decode decision. The next forward can use it.
                confidence=torch.softmax(logits.float(),dim=-1).amax(dim=-1).flatten()
                assert confidence.numel()==block_end-block_start
                self.confidence[block_start:block_end].copy_(confidence)
                self.observed_at[block_start:block_end].fill_(self.completed_decodes)
                self.completed_decodes+=1
            return result
        decoder.decode=decode

    def finish(self,outdir,mode,token_ids):
        super().finish(outdir,mode,token_ids)
        if mode!='trace':
            return
        assert self.completed_decodes>0,'No completed decode telemetry was observed'
        arrays={}
        metadata=[]
        for row in self.history_rows:
            tag=f"r{row['route_index']}"
            arrays[tag+'_prior_confidence']=row['confidence'].detach().cpu().numpy()
            arrays[tag+'_prior_completed_decode']=row['observed_at'].detach().cpu().numpy()
            valid=arrays[tag+'_prior_completed_decode']>=0
            assert np.all(arrays[tag+'_prior_completed_decode'][valid] < row['completed_decodes_before_dispatch'])
            metadata.append({k:v for k,v in row.items() if k not in ('confidence','observed_at')})
        np.savez_compressed(outdir/f'refinement_history.rank{self.rank}.npz',**arrays)
        write(outdir/f'refinement_history.rank{self.rank}.json',dict(rank=self.rank,rows=metadata,
            completed_decodes=self.completed_decodes,
            feature='Maximum softmax probability from a previously completed decode forward.',
            availability='Snapshot before current dispatch. NaN and step -1 mean no history.',
            limitation='Observer-only extra softmax. Not a calibrated online policy or clean latency result.'))

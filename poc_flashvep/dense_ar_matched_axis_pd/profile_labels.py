"""Attach request labels only to explicit diagnostic Nsight runs."""
import time


def install(cfg,role):
    if not cfg.get('nsys_bounded_capture'):
        return
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    prepare=GPUModelRunner.prepare_inputs
    execute=GPUModelRunner.execute_model
    def prepared(self,*args,**kwargs):
        batch=prepare(self,*args,**kwargs)
        ids=[rid.split('-')[0] for rid in batch.req_ids]
        assert not getattr(self,'_matched_nvtx_open',False)
        torch.cuda.nvtx.range_push('MATCHED_MODEL_SUBMISSION|'+role+
            '|requests='+','.join(ids)+'|queries='+str(batch.num_tokens)+
            '|prefilling='+str(bool(batch.is_prefilling_np[:batch.num_reqs].any()))+
            '|Last='+str(cfg['static_layers'])+'|host='+f'{time.monotonic():.9f}')
        self._matched_nvtx_open=True
        return batch
    def executed(self,*args,**kwargs):
        try:
            return execute(self,*args,**kwargs)
        finally:
            if getattr(self,'_matched_nvtx_open',False):
                torch.cuda.nvtx.range_pop()
                self._matched_nvtx_open=False
    GPUModelRunner.prepare_inputs=prepared
    GPUModelRunner.execute_model=executed

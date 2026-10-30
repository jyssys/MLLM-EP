"""Absolute prompt-state rows and a legal per-request depth frontier barrier."""


def install(cfg):
    from vllm_rlt.worker import buffers,model_runner
    from vllm_rlt.core.scheduler import Scheduler,ScheduledRequest
    from vllm_rlt.request import RequestPhase
    from vllm_rlt.worker.executor import Executor
    if getattr(Scheduler,'_arc_installed',False):return
    Scheduler._arc_installed=True
    maximum=cfg['state_capacity_tokens']
    original_rows=buffers.prefill_stage_rows

    def state_rows(config,scheduler,*,staged_prefill=None):
        original=original_rows(config,scheduler,staged_prefill=staged_prefill)
        return maximum if original else 0

    buffers.prefill_stage_rows=state_rows
    model_runner.prefill_stage_rows=state_rows

    def window_open(self,request):
        return request.num_preluded_tokens<len(request.prompt_token_ids)

    def task(self,request,phase,token_budget,*,start,depth=None):
        n=len(request.prompt_token_ids)
        if n>maximum:raise ValueError('Prompt exceeds the explicit retained-state capacity')
        chunk=self.config.prefill_chunk_size
        chunk_start=start//chunk*chunk
        end=min(n,chunk_start+chunk)
        if phase==RequestPhase.PREFILLING_RECURRENT:
            if depth is None:raise ValueError('A recurrent task needs its depth')
            cut=getattr(request,'arc_export_cut',self.total_ut_steps)
            if depth>=cut:return None
            left,right=request.prefill_depth_ranges[depth]
            if left!=start:raise RuntimeError('A recurrent range has a stale frontier')
            end=min(end,right,request.num_preluded_tokens)
        count=min(token_budget,end-start)
        if count<=0:return None
        return ScheduledRequest(request,start,count,depth=depth,chunk_start=chunk_start,
                                prefill_hidden_row_offset=start,
                                completes_prompt=phase==RequestPhase.PREFILLING_RECURRENT and
                                depth==self.total_ut_steps-1 and start+count==n)

    Scheduler._prelude_window_open=window_open
    Scheduler._make_wavefront_task=task
    submit=Executor.submit

    def no_mixed(self,scheduled):
        phases={scheduled.phase_for(r) for r in scheduled.requests}
        if RequestPhase.RECURRENT in phases and RequestPhase.PREFILLING_RECURRENT in phases:
            raise RuntimeError('Forbidden Prefill/Decode GEMM co-batching')
        return submit(self,scheduled)

    Executor.submit=no_mixed

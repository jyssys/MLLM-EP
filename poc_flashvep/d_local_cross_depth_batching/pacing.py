"""Reuse the observed-output slice credit without changing native frontiers."""
import os
from .common import ROOT, write


def install(cfg):
    if not cfg.get('dlocal_output_credit'):
        return
    from ..fluidtoken_loop_split.probe_quantum import OutputCredit
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.request import RequestPhase as P
    initialize,ready,submit,closed=StageEngine.__init__,StageEngine._execution_ready,ModelRunner.submit,StageEngine.close

    def background(engine):
        return any(r.phase in (P.PRELUDE,P.RECURRENT,P.CODA) for r in engine.scheduler.requests.values())

    def initialized(engine,*args,**kwargs):
        value=initialize(engine,*args,**kwargs)
        if engine.stage_config.role!='decode':
            return value
        engine._dlocal_output_credit=OutputCredit()
        engine.model_runner._dlocal_credit_engine=engine
        step=engine.step
        seen={}
        def stepped():
            outputs=step()
            advanced=False
            for output in outputs:
                count=len(output.token_ids)
                if count>seen.get(output.request_id,0):
                    advanced=True;seen[output.request_id]=count
                if output.finished:
                    seen.pop(output.request_id,None)
            if advanced:
                engine._dlocal_output_credit.refresh()
            return outputs
        engine.step=stepped
        return value

    def eligible(engine,request):
        accepted=ready(engine,request)
        credit=getattr(engine,'_dlocal_output_credit',None)
        if credit is None or request.phase not in (P.PREFILLING_PRELUDE,P.PREFILLING_RECURRENT):
            return accepted
        return accepted and credit.allows(background(engine))

    def submitted(runner,scheduled_batch=None,*,batch):
        engine=getattr(runner,'_dlocal_credit_engine',None)
        if engine is not None and any(s.chunk_start is not None for s in batch.spans):
            assert batch.loop_count<=cfg.get('dlocal_quantum',1),'Output-credit slice exceeded its loop bound'
            engine._dlocal_output_credit.consume(background(engine))
        return submit(runner,scheduled_batch,batch=batch)

    def close(engine):
        credit=getattr(engine,'_dlocal_output_credit',None)
        if credit is not None:
            write(ROOT/'raw'/cfg['name']/f'dlocal_output_credit_{os.getpid()}.json',dict(vars(credit),
                scope='At most one bounded prefill submission per observed output step while decode is active. Statistical SLO screen, not a deadline guarantee.'))
        return closed(engine)
    StageEngine.__init__,StageEngine._execution_ready,StageEngine.close=initialized,eligible,close
    ModelRunner.submit=submitted

"""Apply the same legal D loop quantum to Token and Loop continuations."""
import os
from .common import ROOT,write


def install(cfg):
    quantum=cfg.get('idle_D_loop_quantum')
    if quantum is None:return
    assert quantum>=1
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import RequestPhase as P
    initialize=StageEngine.__init__
    def initialized(engine,*args,**kwargs):
        result=initialize(engine,*args,**kwargs)
        if engine.stage_config.role!='decode':return result
        planned=engine.scheduler._planned_loop_count
        def bounded(items):
            count=planned(items)
            if any(item.phase==P.PREFILLING_PRELUDE for item in items):return 0
            if any(item.phase==P.PREFILLING_RECURRENT for item in items):return min(count,quantum)
            return count
        engine.scheduler._planned_loop_count=bounded
        write(ROOT/'raw'/cfg['name']/f'D_quantum_contract_{os.getpid()}.json',dict(quantum=quantum,
            scope='Same legal recurrent boundary quantum for Token and Loop. Prelude stays separate. No kernel preemption or adaptive admission is implemented.'))
        return result
    StageEngine.__init__=initialized

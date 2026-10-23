"""CPU submission accounting. No CUDA timing or tensor read occurs here."""
import time

class WorkLedger:
    def __init__(self,engine):
        self.engine=engine;self.original=engine.executor.submit
        self.rows=[];engine.executor.submit=self.submit

    def submit(self,batch):
        spans=[]
        for r in batch.requests:
            phase=batch.phase_for(r).value
            if phase.startswith('prefill'):
                spans.append(dict(id=r.request.request_id,start=r.token_start,tokens=r.token_count,
                    depth=r.depth or 0,G=4 if phase=='prefill' else r.loop_count if phase=='prefilling_recurrent' else 0,
                    phase=phase))
        value=self.original(batch)
        if spans:self.rows.append(dict(host_return_s=time.perf_counter(),spans=spans))
        return value

    def close(self):
        if self.engine is None:return
        self.engine.executor.submit=self.original
        self.engine=None;self.original=None

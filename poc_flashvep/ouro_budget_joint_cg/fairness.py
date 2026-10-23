"""Credit native coda delivery, never merely submitted recurrent work."""
import time

PREFILL=('prefill','prefilling_prelude','prefilling_recurrent')
DECODE=('prelude','recurrent','coda')

class CompletionFairness:
    def __init__(self,native,enabled=True,max_decode_turns=32):
        self.native=native
        self.enabled=enabled
        self.max_decode_turns=max_decode_turns
        self.targets=[]
        self.turns=0
        self.events=[]

    def __getattr__(self,key):
        return getattr(self.native,key)

    def record_batch(self,phase):
        self.native.record_batch(phase)

    def pending(self,scheduler):
        if not self.targets:
            return False
        delivered=[(r,n) for r,n in self.targets if len(r.generated_token_ids)>n]
        live=[(r,n) for r,n in self.targets if scheduler.requests.get(r.request_id) is r]
        if delivered:
            self.events.append(dict(kind='NATIVE_CODA_HOST_DELIVERY',host_s=time.perf_counter(),
                ids=[r.request_id for r,n in delivered],issued_decode_turns=self.turns))
            self.targets=[];self.turns=0
            return False
        if not live:
            self.events.append(dict(kind='DEBT_REMOVED_NO_LIVE_TARGET',host_s=time.perf_counter()))
            self.targets=[];self.turns=0
            return False
        self.targets=live
        return True

    def arm(self,scheduler,batch):
        if not any(batch.phase_for(r).value in PREFILL for r in batch.requests):
            return
        self.targets=[(r,len(r.generated_token_ids)) for r in scheduler.requests.values()
            if r.phase.value in DECODE and len(r.generated_token_ids)>0]
        self.turns=0
        if self.targets:
            self.events.append(dict(kind='PREFILL_DEBT_ARMED',host_s=time.perf_counter(),
                ids=[r.request_id for r,n in self.targets],token_counts=[n for r,n in self.targets]))

    def schedule(self,scheduler,*,prefer_recurrent=False):
        if not self.enabled:
            return self.native.schedule(scheduler,prefer_recurrent=prefer_recurrent)
        from vllm_rlt.request import RequestPhase
        if self.pending(scheduler):
            if self.turns>=self.max_decode_turns:
                self.events.append(dict(kind='BOUNDED_DECODE_ESCAPE_NOT_COMPLETION_CREDIT',host_s=time.perf_counter(),turns=self.turns))
                self.targets=[];self.turns=0
            else:
                for phase in (RequestPhase.CODA,RequestPhase.PRELUDE,RequestPhase.RECURRENT):
                    if scheduler.has_ready(phase):
                        result=scheduler._take(phase)
                        if result is not None:
                            self.turns+=1
                            return result
                # Native async engine waits for its pending coda readback.
                # It must not manufacture a completion or consume unready KV.
                return None
        result=self.native.schedule(scheduler,prefer_recurrent=prefer_recurrent)
        if result is not None:
            self.arm(scheduler,result)
        return result

def attach(engine,enabled=True,max_decode_turns=32):
    wrapper=CompletionFairness(engine.scheduler.policy,enabled,max_decode_turns)
    engine.scheduler.policy=wrapper
    return wrapper

"""Past-output feedback only. A service budget is not a future deadline."""
from collections import deque
import numpy as np

class BudgetController:
    def __init__(self,mode='FIXED',initial_ms=80.,target_ITL_ms=270.,minimum_dwell=16):
        self.mode=mode;self.value=float(initial_ms);self.target=float(target_ITL_ms)
        self.menu=(10.,20.,40.,60.,80.,120.,160.,240.)
        self.minimum_dwell=minimum_dwell;self.since_update=0;self.gaps=deque(maxlen=64);self.last={};self.updates=[]

    def observe_output(self,request_id,observed_s,prefill_waiting=False,decode_pressure=0):
        previous=self.last.get(request_id);self.last[request_id]=observed_s
        if previous is not None:
            self.gaps.append(1000*(observed_s-previous));self.since_update+=1
        if self.mode=='FIXED' or self.since_update<self.minimum_dwell:
            return self.value
        old=self.value
        if self.mode=='PRESSURE_STYLE':
            self.value=40. if prefill_waiting and decode_pressure>1 else 160.
        elif self.mode=='FEEDBACK':
            if len(self.gaps)<32:
                return self.value
            tail=float(np.quantile(self.gaps,.95))
            index=min(range(len(self.menu)),key=lambda i:abs(self.menu[i]-self.value))
            if tail>1.15*self.target:
                index=max(0,index-1)
            elif tail<.85*self.target and prefill_waiting:
                index=min(len(self.menu)-1,index+1)
            self.value=self.menu[index]
        else:
            raise ValueError(self.mode)
        self.since_update=0
        self.updates.append(dict(request_id=request_id,observed_s=observed_s,old_ms=old,new_ms=self.value,
            observations=len(self.gaps),mode=self.mode))
        return self.value

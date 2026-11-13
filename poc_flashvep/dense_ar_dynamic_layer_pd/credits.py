"""Track prospective KV credits without replacing the native KV manager."""
from dataclasses import dataclass,field


@dataclass
class Reservation:
    generation:str
    cut:int
    blocks:dict
    committed:bool=False
    allocated:set=field(default_factory=set)


class Credits:
    def __init__(self):self.reservations={};self.internal={};self.history=[]
    def map_request(self,internal,external):self.internal[internal]=external
    def outstanding(self,exclude=None):
        return sum(n for r in self.reservations.values() for rid,n in r.blocks.items()
            if rid not in r.allocated and rid!=exclude)
    def reserve(self,generation,cut,requests,blocks_each,free,decode_reserve=0,max_pending=16):
        assert generation not in self.reservations and cut in (24,28,32,34)
        assert requests and len(set(requests))==len(requests) and blocks_each>0
        live={rid for r in self.reservations.values() for rid in r.blocks}
        assert not live.intersection(requests),'A request cannot own two reservations'
        pending=sum(len(r.blocks)-len(r.allocated) for r in self.reservations.values())
        if free<self.outstanding()+decode_reserve+blocks_each*len(requests):return False
        if pending+len(requests)>max_pending:return False
        self.reservations[generation]=Reservation(generation,cut,dict.fromkeys(requests,blocks_each))
        self.history.append(dict(action='RESERVE',generation=generation,cut=cut,requests=list(requests)))
        return True
    def commit(self,generation):
        r=self.reservations[generation];assert not r.committed;r.committed=True
        self.history.append(dict(action='COMMIT',generation=generation))
    def cancel(self,generation):
        r=self.reservations[generation];assert not r.committed and not r.allocated
        del self.reservations[generation];self.history.append(dict(action='CANCEL_BEFORE_PREFIX',generation=generation))
    def validate(self,generation,request,cut):
        r=self.reservations[generation]
        assert r.committed and r.cut==cut and request in r.blocks,'Stale or mismatched committed cut'
    def allocated(self,internal):
        external=self.internal.get(internal)
        for r in self.reservations.values():
            if external in r.blocks:
                assert r.committed
                r.allocated.add(external)
    def finish(self,external):
        for generation,r in list(self.reservations.items()):
            if external not in r.blocks:continue
            assert external in r.allocated and r.committed
            del r.blocks[external];r.allocated.remove(external)
            self.history.append(dict(action='FINISH',generation=generation,request=external))
            if not r.blocks:del self.reservations[generation]


LEDGER=Credits()

"""Causal no-refill barrier. Successor order never uses depth or future work."""


class NoRefillWave:
    def __init__(self,ids):
        self.order={rid:i for i,rid in enumerate(ids)}
        self.expected=len(ids);self.active=set(ids);self.finished=set();self.pending={}

    def complete(self,rid,successor):
        assert rid in self.active,'A retired request cannot complete another token wave'
        assert rid not in self.finished,'Duplicate completion in one token wave'
        assert len(self.finished)<self.expected
        self.finished.add(rid)
        if successor is not None:self.pending[rid]=successor
        if len(self.finished)<self.expected:return []
        ready=[self.pending[rid] for rid in sorted(self.pending,key=self.order.__getitem__)]
        self.expected=len(ready);self.active=set(self.pending);self.finished.clear();self.pending.clear()
        return ready

"""Small full-ID observer for exact count-matrix diagnostics. No weights are copied."""
from collections import Counter
import numpy as np

class FullRoutes:
    def __init__(self,a):
        self.a=a;self.calls=Counter();self.values={};self.handles=[]
        self.handles.append(a.model.register_forward_pre_hook(self.begin))
        for site,block in a.sites.items():self.handles.append(block.router.register_forward_hook(self.hook(site)))

    def begin(self,module,inputs):
        self.calls.clear();self.values.clear()

    def hook(self,site):
        def collect(module,inputs,outputs):
            ids=outputs[1] if self.a.name.startswith('ifm') else outputs[3]
            loop=self.calls[site];self.calls[site]+=1
            assert int(ids.max())<256
            self.values[f'{site}.r{loop}']=ids.detach().cpu().numpy().astype(np.uint8)
        return collect

    def close(self):
        for h in self.handles:h.remove()
        for site in self.a.sites:
            expected=self.a.loops if site in self.a.recurrent else 1
            assert self.calls[site]==expected

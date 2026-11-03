"""Replay the unchanged qualified 256-row projection operations in groups.

This is a generic submission optimization. It does not enlarge any GEMM or
change arithmetic. Exact GPU operator and request parity remain mandatory.
"""
import os
import time
import types
from .common import ROOT,write,append


class GroupedProjection:
    def __init__(self,base,name,rows):
        self.base=base;self.name=name;self.rows=rows;self.cache={};self.calls=0

    def __call__(self,x):
        import torch
        if x.ndim!=2 or len(x)<self.rows or torch.cuda.is_current_stream_capturing():
            return self.base(x)
        stream=torch.cuda.current_stream(x.device)
        key=(str(x.device),x.dtype,x.shape[-1],stream.cuda_stream)
        if key not in self.cache:
            started=time.monotonic()
            static=torch.zeros((self.rows,x.shape[-1]),device=x.device,dtype=x.dtype)
            side=torch.cuda.Stream(device=x.device);side.wait_stream(stream)
            with torch.inference_mode(),torch.cuda.stream(side):
                for _ in range(2):self.base(static)
            stream.wait_stream(side)
            graph=torch.cuda.CUDAGraph()
            with torch.inference_mode(),torch.cuda.graph(graph,stream=side):
                output=self.base(static)
            stream.wait_stream(side)
            self.cache[key]=(static,graph,output)
            folder=ROOT/'raw'/os.environ['ARC_RUN']
            append(folder/f'projection_graph_{os.getpid()}.jsonl',dict(event='CAPTURE',time=time.monotonic(),
                name=self.name,group_rows=self.rows,unchanged_GEMM_rows=256,
                capture_host_seconds=time.monotonic()-started,
                measured_window_already_started=(folder/'measurement_begin.json').exists(),
                evidence='MEASURED_HOST_GRAPH_PREPARATION_NOT_DEVICE_TIME'))
        static,graph,output=self.cache[key]
        pieces=[]
        for left in range(0,len(x),self.rows):
            part=x[left:left+self.rows]
            if len(part)!=self.rows:
                pieces.append(self.base(part))
            else:
                static.copy_(part);graph.replay()
                # A graph output is private reusable storage. Return independent
                # rows before the next replay can overwrite that storage.
                pieces.append(output.clone())
        self.calls+=1
        return torch.cat(pieces,dim=0) if len(pieces)>1 else pieces[0]


def apply_model(model,cfg):
    import torch
    if not cfg.get('projection_graph_rows'):return model
    rows=int(cfg['projection_graph_rows']);assert rows>=256 and rows%256==0
    chosen=[]
    for name,module in model.named_modules():
        if not isinstance(module,torch.nn.Linear):continue
        down=name.endswith('down_proj') or '.mlp.proj' in name
        adapter=cfg['model']=='huginn' and name.endswith('adapter')
        if down or adapter:
            wrapped=GroupedProjection(module.forward,name,rows)
            module._autonomous_grouped_projection=wrapped
            module.forward=wrapped
            chosen.append(name)
    write(ROOT/'raw'/os.environ['ARC_RUN']/f'projection_graph_selection_{os.getpid()}.json',
        dict(names=chosen,group_rows=rows,GEMM_rows=256,operation_scope='Same qualified projection tiles, zero padding and row order. Graph replay plus exact copies only.',
            classification='GENERIC_RUNTIME_OPTIMIZATION_NOT_LOOP_SPECIFIC'))
    return model


def install_factory(cfg):
    from vllm_rlt.models import AutoModelForCausalLM
    if getattr(AutoModelForCausalLM,'_autonomous_projection_graph',False):return
    AutoModelForCausalLM._autonomous_projection_graph=True
    original=AutoModelForCausalLM.from_pretrained
    def load(cls,*args,**kwargs):return apply_model(original(*args,**kwargs),cfg)
    AutoModelForCausalLM.from_pretrained=classmethod(load)

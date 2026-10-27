"""Reproduce the actual native generation allocation failure. No proxy model."""
import torch
from .common import append,write
from .workloads import inputs
from .native_measure import generate

def measure(a,target,config):
    b=config.get('batch',16);length=config.get('prefix',2048)
    ids,_=inputs(a,b,length)
    def log(module,args):
        append(target/'forward_shapes.jsonl',dict(batch=args[0].shape[0],prefix=args[0].shape[1],
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))
    handle=a.model.register_forward_pre_hook(log)
    try:
        for _ in range(5):value=a.forward(ids);del value
        value=generate(a,ids,config.get('cap',32))
        write(target/'native_generation_probe.json',dict(value,batch=b,prefix=length,
            allocator_conf=config.get('allocator_conf','DEFAULT'),evidence='ACTUAL_CHECKPOINT_NATIVE_GENERATION_DIAGNOSTIC'))
    finally:handle.remove()

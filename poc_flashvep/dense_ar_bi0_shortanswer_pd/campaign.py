"""Dispatch a BI0 trace with explicit accuracy and measurement scope."""
import json
import numpy as np
from .common import ROOT,save
from .qualification import configuration
from .run import execute
from .metrics import summarize
from .quality import quality_gate


def serving(name,method,dataset,count,rate,seed,**extra):
    offsets=np.cumsum(np.random.default_rng(seed).exponential(1/rate,count));offsets-=offsets[0]
    cfg=configuration(name,method,dataset,count=count,serial=False,state_capture=False,
        correctness_only=False,batch_invariant=False,arrival_offsets=offsets.tolist(),
        arrival_seed=seed,offered_QPS=rate,wall_timeout=3600,serving_timeout=2700,**extra)
    assert execute(save(cfg)),name
    result=summarize(name)
    result['quality_gate']=quality_gate(name)
    return result

"""Qualify standard continuous workers before any Main performance sweep."""
import json
import time
from .common import ROOT,write,digest,save,initialize
from .run import execute


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def configuration(name,method,dataset,count=8,**overrides):
    value=dict(name=name,method=method,entry_module='poc_flashvep.dense_ar_bi0_shortanswer_pd.serving',
        gpus=[0,1],dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),count=count,
        token_budget=8192,max_num_seqs=1024,batch_invariant=False,kv_layout='HND',output_cap=2048,
        arrival_offsets=[0.]*count,serial=True,state_capture=True,correctness_only=True,
        protocol='OFFICIAL_PROXY',wall_timeout=1800,serving_timeout=1200)
    value.update(overrides);return value



"""Create explicit run configurations and preserve paired arrival traces."""
import json
import numpy as np
from .common import ROOT,digest
from .run import save


def config(name,method,dataset,cut=0,pair=(0,1),qps=1.,seed=2026112101,
           count=None,output=2048,fixed=False,controlled=False,correctness=False,**extra):
    path=ROOT/'data'/dataset;data=json.loads(path.read_text())
    n=len(data) if count is None else count;assert 0<n<=len(data)
    offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(1/qps,n-1))].tolist()
    value=dict(name=name,method=method,gpus=list(pair),batch_invariant=False,dataset=dataset,
        dataset_SHA256=digest(path),count=n,output_cap=output,ignore_eos=fixed,
        token_budget=8192,max_num_seqs=1024,arrival_offsets=offsets,qps=qps,seed=seed,
        protocol='OFFICIAL_PROXY',execution_family='MATCHED_NO_COMPILE',async_scheduling=True,
        KV_block_tokens=128,restore_official_UCX_init=True,nixl_num_threads=4,
        controlled_pipeline=controlled,source_credit_limit=4,correctness_only=correctness,
        axis_state_capture=correctness,state_capture=False,
        hidden_source_slots=4,hidden_destination_slots=4,serving_timeout=3600,wall_timeout=4500,
        quality_contract='BI0. Strict tokens for identical geometry. Task metrics and state tolerances for different geometry.')
    if method in ('Layer','Token','Hybrid'):value.update(axis_kind=method,axis_cut=cut)
    if not controlled and method in ('Native','Layer','Token','Hybrid'):value['continuous_source_credit_limit']=4
    value.update(extra)
    if value.get('reference_single') and 'continuous_source_credit_limit' not in extra:
        value['continuous_source_credit_limit']=0
    return value


def create(*a,**kw):return save(config(*a,**kw))

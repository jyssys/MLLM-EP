"""Use exact published inference and freeze each new controlled experiment."""
import json
import shutil
from .common import ROOT,PREVIOUS,bind_legacy,digest,initialize


def prepare_data():
    initialize()
    for relative in ('data/pool_contract.json','data/discovery72_declared_cycle.json',
                     'raw/HISTORICAL_original_outputs/requests.jsonl'):
        source=PREVIOUS/relative;target=ROOT/relative
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        assert digest(source)==digest(target)


def base(name,method,layers=4,count=72,cap=16,qps=2.,seed=2026111900,batch=1,window=32,**changes):
    prepare_data();bind_legacy()
    from ..dense_ar_matched_axis_pd.configuration import base as previous
    from ..dense_ar_layer_token_pd.token_axis import matched_token_plan
    requested=4 if method=='Token' and layers==2 else layers
    cfg=previous(name,method,requested,count,cap,qps,seed,'matched144.json',
        phase='AXIS_SWEETSPOT_CONTROL',reference_name='HISTORICAL_original_outputs',
        common_prospective_KV_reservation=True,execution_contract_version=3)
    cfg.update(static_layers=layers if method in ('Layer','Token') else 0,
        layer_cut=36-layers if method in ('Layer','Token') else cfg['layer_cut'],
        P_group_size=1,D_admission_batch=batch,source_outstanding_limit=window,
        batch_matched_control=True,matched_request_batch=1,
        matrix_stage='discovery',warmup_count=2,
        warmup_layer_counts=[layers if method=='Layer' else 0],
        wall_timeout=1200,timeout=900,startup_timeout=120)
    if method=='Token':
        cfg['token_suffix']={2:448,4:896,8:1792,12:2672}[layers]
        if count%9==0:
            cfg['token_suffix_by_request']=matched_token_plan(layers,count,group_size=1)
        else:
            cfg['token_suffix_by_request']=[cfg['token_suffix']]*count
    if batch>1:
        assert count%batch==0
        cfg.update(matched_D_cohort_size=batch,cohort_runtime_version=3,
            entry_module='poc_flashvep.dense_ar_matched_axis_pd.fixed_matched_cohort_v3')
    cfg.update(changes)
    assert cfg['P_group_size']==1 and cfg['P_slots']==cfg['D_slots']==1024
    assert cfg['token_budget']==cfg['D_token_budget']==8192
    return cfg


def state(method,layers,count):
    prepare_data();bind_legacy()
    from ..dense_ar_matched_axis_pd.prepare import qualification
    requested=4 if method=='Token' and layers==2 else layers
    cfg=qualification(method,count,requested,version=20261119)
    version=2
    cfg.update(name=f'state_{method}_Last{layers}_N{count}_B1_v{version}',
        static_layers=layers if method!='Native' else 0,layer_cut=36-layers if method!='Native' else cfg['layer_cut'],
        P_group_size=1,D_admission_batch=1,source_outstanding_limit=32,
        batch_matched_control=True,matched_request_batch=1,matrix_stage='correctness',
        witness_hidden_partition_tokens=[432,448,880,896,1776,1792,2656,2672],startup_timeout=120)
    if method=='Token' and layers==2:
        cfg.update(token_suffix=448,token_suffix_by_request=[(432,448)[i%2] for i in range(count)])
    return cfg

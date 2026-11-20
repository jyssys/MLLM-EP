"""Run paired BI0 cells under a fixed resource and numerical contract."""
import json
import time
import numpy as np
from .common import ROOT,save,write,activate
from .run import execute

activate()
from ..dense_ar_bi0_shortanswer_pd import run as inherited_run
inherited_run.execute=execute
from ..dense_ar_bi0_shortanswer_pd.qualification import configuration
from ..dense_ar_bi0_shortanswer_pd.metrics import summarize
from ..dense_ar_bi0_shortanswer_pd.quality import quality_gate
from ..dense_ar_bi0_shortanswer_pd.axis_timeline import analyze as axis_analyze
from ..dense_ar_bi0_shortanswer_pd.serving_analysis import analyze as serving_analyze


def cell(name,axis='Native',cut=0,rate=2.35,seed=2026112071,credit=4,
         dataset='longbench_shortanswer200_v2.json',count=200,**overrides):
    offsets=np.cumsum(np.random.default_rng(seed).exponential(1/rate,count))
    offsets-=offsets[0]
    extra=dict(controlled_pipeline=True,source_credit_limit=credit,
        execution_family='MATCHED_NO_COMPILE',async_scheduling=True,
        KV_block_tokens=128,restore_official_UCX_init=True,quiescent_witness=True,
        credit_event_instrumentation=True,phase='AXIS_ROOTCAUSE_CONTROLLED',axis_mode='STATIC')
    if cut:
        extra.update(axis_kind=axis,axis_cut=cut,axis_state_capture=False,hidden_source_slots=4,
            qualified_controlled_performance=True,runtime_qualification_tag='fast128_v3',
            token_teacher_gate_tag='fast128_v3',burst_gate_tag='fast128_v3',qualified_async_runtime=True)
    extra.update(overrides)
    settings=dict(count=count,serial=False,state_capture=False,correctness_only=False,
        batch_invariant=False,arrival_offsets=offsets.tolist(),arrival_seed=seed,
        offered_QPS=rate,wall_timeout=3600,serving_timeout=2700)
    settings.update(extra)
    cfg=configuration(name,axis,dataset,**settings)
    if not execute(save(cfg)):
        return dict(name=name,axis=axis,cut=cut,QPS=rate,seed=seed,credit=credit,status='FAILED')
    metrics=summarize(name)
    quality=quality_gate(name) if not cfg.get('diagnostic_controlled_output_replay') else dict(
        status='NOT_APPLICABLE_FIXED_OUTPUT_DIAGNOSTIC',CI_passed=False,
        scope='This causal workload fixes output counts and ignores EOS. It is not a Natural-generation quality arm.')
    metrics['quality_gate']=quality
    trace=serving_analyze(name)
    geometry=axis_analyze(name) if axis!='Agg' else None
    if geometry is not None:assert geometry['all_prompt_work_accounted'],name
    if cfg['controlled_pipeline']:
        assert all(set(map(int,r['actual_request_batch_histogram']))=={1}
                   for r in trace['geometry'].values()),name
    return dict(name=name,axis=axis,cut=cut,QPS=rate,seed=seed,credit=credit,
        status='CORRECTNESS_ONLY' if cfg.get('correctness_only') else 'MEASURED_ONLINE',metrics=metrics,quality=quality,
        actual_geometry=trace['geometry'],prompt_work_accounted=geometry['all_prompt_work_accounted'] if geometry else None)


def smoke():
    from ..dense_ar_bi0_shortanswer_pd.axis_qualification import main
    main(tag='axis_rootcause_overlay_v2',cuts=(12,),block=128,restore=True,asynchronous=True)
    gate=json.loads((ROOT/'analysis/BI0_axis_state_qualification_axis_rootcause_overlay_v2.json').read_text())
    assert len(gate['results'])==2 and all(r['passed'] for r in gate['results'])
    write(ROOT/'state/new_overlay_runtime_smoke_complete.json',dict(time=time.time(),passed=True,
        scope='New namespace and coordinator. Original worker and math paths are unchanged. New N8 state/lifetime tests cover cut12. Other cuts retain explicitly inherited gates.'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true',required=True)
    p.parse_args();smoke()

"""Revisit the published winner without changing an active experiment."""
import hashlib
import json
import time
import numpy as np
from .common import ROOT,REPO,write,digest,initialize
from .run import execute
from ..dense_ar_bi0_shortanswer_pd.qualification import configuration,records
from ..dense_ar_bi0_shortanswer_pd.metrics import summarize
from ..dense_ar_bi0_shortanswer_pd.serving_analysis import analyze as geometry
from ..dense_ar_bi0_shortanswer_pd.axis_timeline import analyze as coverage

HISTORICAL=REPO/'poc_flashvep/reports/dense_ar_axis_sweetspot_knob_discovery_20261119'
SEEDS=(2026111920,2026111921,2026111922)


def prepare():
    initialize()
    source=HISTORICAL/'data/matched144.json'
    original=json.loads(source.read_text())[:72]
    rows=[]
    for row in original:
        ids=row['input_ids']['qwen8'];assert len(ids)==8000
        rows.append(dict(row,input_tokens=len(ids),task='HISTORICAL_CONTROLLED_INPUT8000',
            input_ids_SHA256=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
            output_cap=16,ignore_eos=True))
    for filename,values in (('historical_input8000_N72.json',rows),('historical_input8000_N8.json',rows[:8])):
        path=ROOT/'data'/filename
        if path.exists():assert json.loads(path.read_text())==values
        else:write(path,values)
    design=ROOT/'configs/frozen_historical_bridge_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),source_commit='4595080c',historical_source=str(source),
            historical_source_SHA256=digest(source),N=72,cut=8,input=8000,output=16,
            QPS=6,seeds=list(SEEDS),actual_batch=[1,1,1],B=8192,S=1024,KV_block=128,
            source_credit=32,hidden_source_slots=32,context_limit=8016,async_scheduling=True,
            bridge_A='Reuse all three published Historical Runtime/BI1 paired starts. Verify raw hashes and configuration. No new A measurement is claimed.',
            bridge_B='Current runtime BI1. Fresh N8 state/output gates. Initially one paired start. Timings stay DIAGNOSTIC because BI1 online qualification is not inherited from BI0.',
            bridge_C='Current runtime BI0. Three paired starts and matching Native references. Existing BI0 runtime gates plus fresh workload state gates apply.',
            bridge_D='Reuse the separate current BI0 ShortAnswer200/c4 family. It is not an input-only or numerical-flag-only intervention.',
            numerical_AB='Current B/C configuration is identical except numerical flag and evidence labels. Historical A/B also changes runtime, transport, block layout and credit semantics.',
            work='Historical uses an exact aggregate mixed-suffix plan. Current code keeps its unchanged nearest16 prefix rule. Record the resulting row difference. Do not call A/B an exact-work intervention.',
            deadline='No new job in the reserved closeout hour. Remaining cells are NOT_RUN. No active GPU job is interrupted.'))
    return rows


def save(cfg):
    # This is the sole explicitly labeled BI1 diagnostic branch of the BI0 campaign.
    assert cfg.get('historical_numeric_bridge')
    if cfg.get('axis_kind'):assert cfg['axis_cut']==8
    assert cfg['batch_invariant'] is False or cfg.get('correctness_only')
    cfg['entry_module']='poc_flashvep.dense_ar_bi0_axis_rootcause.serving'
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg
    else:write(path,cfg)
    return cfg['name']


def settings(BI):
    return dict(batch_invariant=bool(BI),historical_numeric_bridge=True,
        execution_family='MATCHED_NO_COMPILE',async_scheduling=True,KV_block_tokens=128,
        restore_official_UCX_init=True,nixl_num_threads=4,context_limit=8016,output_cap=16,
        ignore_eos=True,diagnostic_controlled_output_replay=True,
        replay_output_lengths=[16]*72,hidden_source_slots=32,credit_event_instrumentation=True,
        quiescent_witness=True,phase='HISTORICAL_INPUT8000_LAST8_RUNTIME_BRIDGE')


def qualify(BI):
    import torch
    torch.set_num_threads(1)
    label=f'BI{int(BI)}';dataset='historical_input8000_N8.json'
    common=dict(settings(BI),count=8,serial=True,state_capture=False,correctness_only=True,
        async_qualification_only=True,numerical_probe=True,numerical_probe_sync=True)
    names={key:f'HistBridgeGate_{label}_{key}_N8_v1' for key in ('Reference','Layer','Token','TokenTeacher')}
    cfg=configuration(names['Reference'],'Native',dataset,reference_single=True,gpus=[1],**dict(common,state_capture=True))
    assert execute(save(cfg)),names['Reference']
    for axis in ('Layer','Token'):
        cfg=configuration(names[axis],axis,dataset,axis_kind=axis,axis_cut=8,**common)
        assert execute(save(cfg)),names[axis]
    cfg=configuration(names['TokenTeacher'],'Native',dataset,reference_single=True,gpus=[1],
        token_teacher_cut=8,**dict(common,state_capture=True))
    assert execute(save(cfg)),names['TokenTeacher']
    results=[]
    for axis in ('Layer','Token'):
        actual=ROOT/'raw'/names[axis];ref=ROOT/'raw'/names['Reference' if axis=='Layer' else 'TokenTeacher']
        reference=json.loads((ref/'prompt_state_single.json').read_text())['requests']
        p=json.loads((actual/'axis_events_prefill.json').read_text())['states']
        d=json.loads((actual/'axis_events_decode.json').read_text())['states']
        expected={r['sample_id']:r['output_ids'] for r in records(ref/'requests.jsonl')}
        observed={r['sample_id']:r['output_ids'] for r in records(actual/'requests.jsonl')}
        queries=json.loads((ref/'actual_queries_single.json').read_text())['rows']
        split=[q for role in ('prefill','decode') for q in json.loads((actual/f'actual_queries_{role}.json').read_text())['rows']]
        checks=[]
        for i in range(8):
            rid=f'q{i}';hidden=torch.load(ref/f'numerical_single_{rid}.pt',weights_only=True)['hidden_last']
            h=hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            spans=lambda values:[(q['start'],min(q['count'],8000-q['start'])) for q in values if q['request_id']==rid and q['start']<8000]
            checks.append(dict(request_id=rid,imported_KV_exact=p[rid]['KV']==d['import_'+rid]['KV'],
                complete_KV_exact=reference[rid]['KV']==d[rid]['KV'],hidden_exact=h==d[rid]['last_hidden_SHA256'],
                output_exact=expected[i]==observed[i],query_geometry_exact=axis=='Layer' or spans(queries)==spans(split)))
        lifetime={p.name:json.loads(p.read_text())['passed'] for p in actual.glob('*lifetime*.json')}
        passed=all(all(v for k,v in row.items() if k!='request_id') for row in checks) and bool(lifetime) and all(lifetime.values())
        results.append(dict(axis=axis,passed=passed,checks=checks,lifetime=lifetime,names=names))
    gate=dict(BI=bool(BI),passed=all(r['passed'] for r in results),results=results,evidence='CORRECTNESS_ONLY')
    write(ROOT/'analysis'/f'historical_workload_state_gate_{label}.json',gate)
    return gate['passed']


def performance(axis,BI,seed,block=128,credit=32):
    assert block in (16,128) and credit in (4,32)
    label=f'BI{int(BI)}';suffix='' if block==128 else '_block16'
    name=f'HistBridgeCurrent_{label}_{axis}8_N72_O16_q6_c{credit}_s{seed}{suffix}_v1'
    budget=json.loads((ROOT/'state/budget.json').read_text())
    if time.time()>budget['last_large_dispatch_epoch']-1200:
        return dict(name=name,axis=axis,BI=bool(BI),seed=seed,status='NOT_RUN',reason='Preserve confirmation drain and closeout time.')
    old_name=f'heldout_confirmation_Layer_Last8_N72_O16_q6_s{seed}_B1_W32_v1'
    old_path=HISTORICAL/'raw'/old_name/'configuration.json'
    if old_path.exists():arrival_offsets=json.loads(old_path.read_text())['arrival_offsets']
    else:
        offsets=np.cumsum(np.random.default_rng(seed).exponential(1/6,72));offsets-=offsets[0]
        arrival_offsets=offsets.tolist()
    extra=dict(settings(BI),count=72,serial=False,state_capture=False,correctness_only=bool(BI),
        controlled_pipeline=True,source_credit_limit=credit,KV_block_tokens=block,arrival_offsets=arrival_offsets,
        arrival_seed=seed,offered_QPS=6.,wall_timeout=1200,serving_timeout=900)
    if axis!='Native':
        extra.update(axis_kind=axis,axis_cut=8,axis_state_capture=False,
            qualified_controlled_performance=not BI,runtime_qualification_tag='fast128_v3',
            token_teacher_gate_tag='fast128_v3',burst_gate_tag='fast128_v3',
            qualified_async_runtime=not BI,async_qualification_only=bool(BI))
        if block==16:
            assert not BI,'The block16 bridge is separately gated BI0 only'
            extra.update(qualified_KV16_controlled=True,qualified_controlled_performance=False,
                qualified_async_runtime=False,runtime_qualification_tag='kv16_v1',token_teacher_gate_tag='kv16_v1')
    cfg=configuration(name,axis,'historical_input8000_N72.json',**extra)
    if not execute(save(cfg)):return dict(name=name,axis=axis,BI=bool(BI),seed=seed,status='FAILED')
    metric=summarize(name);shape=geometry(name);work=coverage(name)
    assert all(set(map(int,r['actual_request_batch_histogram']))=={1} for r in shape['geometry'].values())
    assert work['all_prompt_work_accounted']
    return dict(name=name,axis=axis,cut=8,BI=bool(BI),seed=seed,N=72,QPS=6,credit=credit,KV_block=block,
        status='DIAGNOSTIC' if BI else 'MEASURED_ONLINE',metrics=metric,actual_geometry=shape['geometry'],
        scope='Historical fixed-output controlled workload. BI1 is a separately gated numerical diagnostic, not a BI0 primary arm. Source credit32 is not the historical window32 return policy.')


def main():
    prepare();rows=[];gates={}
    # The running GPU job keeps its lease. This controller joins the same queue.
    for BI in (False,True):
        gates[str(int(BI))]=qualify(BI)
        write(ROOT/'analysis/historical_bridge_state_gates_v1.json',dict(gates=gates))
    schedule=[(a,BI,SEEDS[0]) for BI in (False,True) for a in ('Layer','Token')]
    schedule += [('Native',False,SEEDS[0])]
    for index,seed in enumerate(SEEDS[1:]):
        schedule.extend((a,False,seed) for a in (('Token','Layer','Native') if index==0 else ('Native','Layer','Token')))
    for axis,BI,seed in schedule:
        if not gates[str(int(BI))]:
            row=dict(axis=axis,BI=bool(BI),seed=seed,status='NOT_RUN',reason='Fresh workload state gate failed.')
        else:row=performance(axis,BI,seed)
        rows.append(row);write(ROOT/'analysis/historical_runtime_bridge_v1.json',dict(results=rows,gates=gates,
            evidence='MEASURED_ONLINE_AND_DIAGNOSTIC',scope='New current-runtime starts only. Historical A is reused and reported separately. No exact-work or single-runtime-change claim across A/B.'))
        assert row['status']!='FAILED',row
    write(ROOT/'state/historical_runtime_bridge_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/historical_runtime_bridge_hold.json',dict(time=time.time(),error=repr(error)));raise

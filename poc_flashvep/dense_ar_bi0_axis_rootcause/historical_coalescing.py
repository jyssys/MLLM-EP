"""Test one historical transport difference in the unchanged current model path."""
import json
import time
from .common import ROOT,write
from .historical_bridge import settings,save
from .run import execute
from ..dense_ar_bi0_shortanswer_pd.qualification import configuration,records
from ..dense_ar_bi0_shortanswer_pd.metrics import summarize
from ..dense_ar_bi0_shortanswer_pd.axis_timeline import analyze as coverage
from ..dense_ar_bi0_shortanswer_pd.serving_analysis import analyze as geometry


def gate(axis):
    name=f'HistCoalesceGate_BI1_{axis}8_N8_v1'
    extra=dict(settings(True),count=8,serial=True,state_capture=False,correctness_only=True,
        axis_kind=axis,axis_cut=8,numerical_probe=True,numerical_probe_sync=True,
        async_qualification_only=True,nixl_manual_coalesce=True)
    cfg=configuration(name,axis,'historical_input8000_N8.json',**extra)
    if not execute(save(cfg)):return dict(name=name,axis=axis,passed=False,status='FAILED')
    baseline=ROOT/'raw'/f'HistBridgeGate_BI1_{axis}_N8_v1';actual=ROOT/'raw'/name
    p=json.loads((actual/'axis_events_prefill.json').read_text())['states']
    d=json.loads((actual/'axis_events_decode.json').read_text())['states']
    prior=json.loads((baseline/'axis_events_decode.json').read_text())['states']
    expected={r['sample_id']:r['output_ids'] for r in records(baseline/'requests.jsonl')}
    output={r['sample_id']:r['output_ids'] for r in records(actual/'requests.jsonl')}
    rows=[]
    for i in range(8):
        rid=f'q{i}'
        rows.append(dict(request_id=rid,source_import_exact=p[rid]['KV']==d['import_'+rid]['KV'],
            full_KV_matches_no_coalescing=prior[rid]['KV']==d[rid]['KV'],
            final_hidden_matches_no_coalescing=prior[rid]['last_hidden_SHA256']==d[rid]['last_hidden_SHA256'],
            strict_output_matches_no_coalescing=expected[i]==output[i]))
    lifetime={p.name:json.loads(p.read_text())['passed'] for p in actual.glob('*lifetime*.json')}
    passed=all(all(v for k,v in r.items() if k!='request_id') for r in rows) and bool(lifetime) and all(lifetime.values())
    return dict(name=name,axis=axis,passed=passed,rows=rows,lifetime=lifetime,evidence='CORRECTNESS_ONLY')


def main():
    declaration=ROOT/'configs/frozen_historical_coalescing_intervention_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),BI1=True,input=8000,output=16,cut=8,N=72,QPS=6,
            seed=2026111920,actual_batch=[1,1,1],credit=32,hidden_source_slots=32,KV_block=128,
            factor='Only nixl_manual_coalesce changes false to true. Use the already implemented exact-byte helper.',
            gate='Fresh N8 prefix/full KV, hidden, greedy and lifetime equality against the same current BI1 no-coalescing geometry.',
            scope='One paired transport intervention,diagnostic only. No graph,attention,GEMM,ACK policy or numerical flag changes. '
            'Historical canonical-block allocation and other old-runtime differences are not recreated. No extra budget.'))
    while True:
        path=ROOT/'analysis/historical_runtime_bridge_v1.json'
        if path.exists():
            rows=json.loads(path.read_text())['results']
            if all(any(r.get('BI') and r.get('axis')==a and r.get('status')=='DIAGNOSTIC' for r in rows) for a in ('Layer','Token')):break
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    gates=[gate(axis) for axis in ('Layer','Token')]
    write(ROOT/'analysis/historical_coalescing_state_gate_v1.json',dict(results=gates,passed=all(r['passed'] for r in gates)))
    results=[]
    for axis in ('Layer','Token'):
        if not all(r['passed'] for r in gates):row=dict(axis=axis,status='NOT_RUN',reason='Coalescing state gate failed.')
        elif time.time()>json.loads((ROOT/'state/budget.json').read_text())['last_large_dispatch_epoch']-1800:
            row=dict(axis=axis,status='NOT_RUN',reason='Preserve primary confirmation and closeout budget.')
        else:
            old=f'HistBridgeCurrent_BI1_{axis}8_N72_O16_q6_c32_s2026111920_v1'
            cfg=json.loads((ROOT/'configs'/(old+'.json')).read_text())
            cfg.update(name=old+'_coalesced',nixl_manual_coalesce=True,phase='HISTORICAL_COALESCING_ONE_FACTOR_DIAGNOSTIC')
            name=save(cfg)
            if not execute(name):row=dict(name=name,axis=axis,status='FAILED')
            else:
                metric=summarize(name);work=coverage(name);shape=geometry(name)
                assert work['all_prompt_work_accounted']
                row=dict(name=name,axis=axis,status='DIAGNOSTIC',metrics=metric,actual_geometry=shape['geometry'])
        results.append(row);write(ROOT/'analysis/historical_coalescing_intervention_v1.json',dict(results=results,
            evidence='DIAGNOSTIC',scope='Same current BI1 request/arrival/cut/output contract. Source-byte coalescing only. No independent-start CI.'))
        assert row['status']!='FAILED',row
    write(ROOT/'state/historical_coalescing_intervention_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/historical_coalescing_intervention_hold.json',dict(time=time.time(),error=repr(error)));raise

"""Confirm the one-factor block16 reversal, including its Native safety reference."""
import json
import time
from .common import ROOT,write
from .historical_bridge import performance,settings,save,SEEDS
from .run import execute
from ..dense_ar_bi0_shortanswer_pd.qualification import configuration


def native_gate():
    name='HistBridgeGate_BI0_NativePD_block16_N8_v1'
    original=json.loads((ROOT/'configs/frozen_historical_bridge_v1.json').read_text())
    from .historical_bridge import HISTORICAL
    previous=json.loads((HISTORICAL/'raw'/f'heldout_confirmation_Layer_Last8_N72_O16_q6_s{SEEDS[0]}_B1_W32_v1'/'configuration.json').read_text())
    extra=dict(settings(False),count=8,serial=False,state_capture=True,correctness_only=True,
        controlled_pipeline=True,source_credit_limit=32,KV_block_tokens=16,
        arrival_offsets=previous['arrival_offsets'][:8],arrival_seed=SEEDS[0],offered_QPS=6.,
        phase='HISTORICAL_NATIVE_BLOCK16_STATE_ONLY')
    cfg=configuration(name,'Native','historical_input8000_N8.json',**extra)
    if not execute(save(cfg)):return dict(passed=False,name=name,status='FAILED')
    folder=ROOT/'raw'/name
    p=json.loads((folder/'prompt_state_prefill.json').read_text())['requests']
    d=json.loads((folder/'prompt_state_decode.json').read_text())['requests']
    rows=[dict(request_id=f'q{i}',full_import_KV_exact=p[f'q{i}']['KV']==d[f'q{i}']['KV']) for i in range(8)]
    lifetime={p.name:json.loads(p.read_text())['passed'] for p in folder.glob('*lifetime*.json')}
    return dict(passed=all(r['full_import_KV_exact'] for r in rows) and bool(lifetime) and all(lifetime.values()),
        name=name,rows=rows,lifetime=lifetime,evidence='CORRECTNESS_ONLY')


def main():
    while True:
        path=ROOT/'analysis/historical_one_factor_controls_v1.json'
        if path.exists():
            rows=json.loads(path.read_text())['results']
            pair={r['axis']:r for r in rows if r.get('KV_block')==16 and r.get('credit')==32 and r['status']=='MEASURED_ONLINE'}
            if len(pair)==2:break
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    declaration=ROOT/'configs/frozen_historical_block16_confirmation_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),reason='The predeclared block16 one-factor pair showed a Layer TTFT advantage. Confirm, do not select a favorable repeat.',
            N=72,input=8000,output=16,QPS=6,cut=8,seeds=list(SEEDS),actual_batch=[1,1,1],B=8192,S=1024,
            KV_block=16,credit=32,hidden_source_slots=32,BI0=True,
            discovery={axis:dict(name=r['name'],TTFT_p95_ms=r['metrics']['TTFT_p95_ms']) for axis,r in pair.items()},
            rule='Keep all three starts, including losses. Native uses the same block16 and arrival seeds. '
            'Block128 remains the primary main setting. A block16 win is not a best-tuned Layer victory against Token block128. '
            'Current constant1776 suffix has -0.1% D row mismatch. No exact-aggregate work claim.'))
    gate=native_gate();write(ROOT/'analysis/historical_Native_block16_state_gate_v1.json',gate)
    if not gate['passed']:
        write(ROOT/'state/historical_block16_confirmation_complete.json',dict(status='NOT_RUN',reason='Native block16 state gate failed.'));return
    rows=[pair['Layer'],pair['Token']]
    schedule=[('Native',SEEDS[0]),('Token',SEEDS[1]),('Layer',SEEDS[1]),('Native',SEEDS[1]),
        ('Native',SEEDS[2]),('Layer',SEEDS[2]),('Token',SEEDS[2])]
    for axis,seed in schedule:
        row=performance(axis,False,seed,block=16,credit=32);rows.append(row)
        write(ROOT/'analysis/historical_block16_confirmation_v1.json',dict(results=rows,evidence='MEASURED_ONLINE',
            scope='Historical fixed-output72 workload, not current Natural LongBench200. First paired start was discovery. '
            'Two additional independent seeds are confirmation. Report both the all3 interval and held-out2-start limitation.'))
        assert row['status']!='FAILED',row
    write(ROOT/'state/historical_block16_confirmation_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/historical_block16_confirmation_hold.json',dict(time=time.time(),error=repr(error)));raise

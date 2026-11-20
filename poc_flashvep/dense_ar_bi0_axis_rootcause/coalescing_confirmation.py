"""Repeat the transport-only bridge instead of treating one rank flip as causal proof."""
import json
import time
from .common import ROOT,write
from .historical_bridge import HISTORICAL,performance,save
from .run import execute
from ..dense_ar_bi0_shortanswer_pd.metrics import summarize
from ..dense_ar_bi0_shortanswer_pd.axis_timeline import analyze as coverage
from ..dense_ar_bi0_shortanswer_pd.serving_analysis import analyze as geometry


def merged(axis,seed):
    template=f'HistBridgeCurrent_BI1_{axis}8_N72_O16_q6_c32_s2026111920_v1'
    cfg=json.loads((ROOT/'configs'/(template+'.json')).read_text())
    old=f'heldout_confirmation_Layer_Last8_N72_O16_q6_s{seed}_B1_W32_v1'
    arrival=json.loads((HISTORICAL/'raw'/old/'configuration.json').read_text())['arrival_offsets']
    name=f'HistBridgeCurrent_BI1_{axis}8_N72_O16_q6_c32_s{seed}_v1_coalesced'
    cfg.update(name=name,arrival_offsets=arrival,arrival_seed=seed,nixl_manual_coalesce=True,
        phase='HISTORICAL_COALESCING_PAIRED_REPETITION')
    if not execute(save(cfg)):return dict(name=name,axis=axis,seed=seed,status='FAILED',coalesced=True)
    metric=summarize(name);work=coverage(name);shape=geometry(name);assert work['all_prompt_work_accounted']
    return dict(name=name,axis=axis,seed=seed,status='DIAGNOSTIC',coalesced=True,metrics=metric,actual_geometry=shape['geometry'])


def main():
    design=ROOT/'configs/frozen_coalescing_paired_repetition_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),BI1=True,seed20='Reuse the first predeclared paired intervention.',
            new_seeds=[2026111921,2026111922],axes=['Layer','Token'],coalescing=[False,True],
            N=72,input=8000,output=16,cut=8,QPS=6,KV_block=128,credit=32,hidden_source_slots=32,
            actual_batch=[1,1,1],B=8192,S=1024,
            selection='The first transport intervention changed the axis ranking. Repeat both factor levels, not only the favorable treatment.',
            budget='No diagnostic dispatch after22:07 KST. No extension of the original campaign deadline.',
            scope='Mechanism diagnostic,not a BI0 main result. Total3 starts include the original diagnostic. '
            'Only2 new seeds are held out from this decision. Do not call this three held-out confirmations or a decode-protected serving GO.'))
    while not (ROOT/'state/historical_coalescing_intervention_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    gate=json.loads((ROOT/'analysis/historical_coalescing_state_gate_v1.json').read_text())
    original=json.loads((ROOT/'analysis/historical_runtime_bridge_v1.json').read_text())['results']
    rows=[dict(r,coalesced=False) for r in original if r.get('BI') and r['status']=='DIAGNOSTIC']
    treated=json.loads((ROOT/'analysis/historical_coalescing_intervention_v1.json').read_text())['results']
    rows += [dict(r,seed=2026111920,coalesced=True) for r in treated if r['status']=='DIAGNOSTIC']
    for seed,order in ((2026111921,[('Token',True),('Layer',False),('Layer',True),('Token',False)]),
                       (2026111922,[('Token',False),('Layer',True),('Layer',False),('Token',True)])):
        for axis,coalesced in order:
            if not gate['passed'] or time.time()>json.loads((ROOT/'state/budget.json').read_text())['last_large_dispatch_epoch']-1800:
                row=dict(axis=axis,seed=seed,coalesced=coalesced,status='NOT_RUN',reason='Gate or diagnostic budget cutoff.')
            else:
                row=merged(axis,seed) if coalesced else dict(performance(axis,True,seed),coalesced=False)
            rows.append(row)
            write(ROOT/'analysis/coalescing_paired_repetition_v1.json',dict(results=rows,evidence='DIAGNOSTIC',
                scope='Axis x coalescing paired by arrival seed. Keep unfavorable starts. No independent claim from a single favorable treatment. '
                'BI1 numerical flag stays fixed. Native-relative safety for this BI1 transport factorial remains NOT_MEASURED.'))
            assert row['status']!='FAILED',row
    write(ROOT/'state/coalescing_paired_repetition_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/coalescing_paired_repetition_hold.json',dict(time=time.time(),error=repr(error)));raise

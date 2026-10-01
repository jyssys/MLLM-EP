"""Finite restart-blocked physical campaign with atomic resident admission.

Each engine session is capped below 50 waves *before* measurement. Restarts,
warmups and pauses are outside the victim timing windows, identical for all
arms. No failed wave is retried automatically or relabeled as a completed one.
"""
import argparse
import hashlib
from pathlib import Path

from poc_native_lp_ep4.contract import atomic_json, read_json, groups_for
from poc_native_lp_ep4.scripts.analyze import clean_rows, summarize
from poc_native_lp_ep4.scripts import finish_campaign as f

ROOT=Path('poc_native_lp_ep4')


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='v6');a=p.parse_args()
    assert a.version in ('v5','v6');f.VERSION=a.version;v=a.version
    prior=ROOT/'artifacts/runs/release_correctness_v4'
    pilot=ROOT/'artifacts/runs/atomic_admission_correctness_v5'
    for run in (prior,pilot):assert read_json(run/'status.json')['ok']
    for report in ('release_validation_v4.json','atomic_admission_validation_v5.json'):
        assert read_json(ROOT/'artifacts/correctness'/report)['status']=='PASS'
    for source in ('poc_native_lp_ep4/native.py','poc_native_lp_ep4/light_trace.py','poc_native_lp_ep4/contract.py'):
        expected=hashlib.sha256(Path(source).read_bytes()).hexdigest()
        assert all(read_json(run/'source_signature.json')[source]==expected for run in (prior,pilot))
    plan={'version':a.version,'status':'RUNNING','admission':'INSTALLED_PAUSE_KEEP_ENQUEUE_4_PER_DP_RESUME',
        'uniform_resident_prefill':v=='v6',
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'session_wave_cap':48,'reason':'bounded controlled serving, avoid unvalidated long external-DP lifecycle',
        'correctness_runs':[str(prior),str(pilot)],'discovery_runs':[],
        'final_runs':[],'component_runs':[],'extra_component_runs':[],'route_runs':[]}
    path=ROOT/'artifacts'/f'campaign_blocks_{v}.json'
    def checkpoint():atomic_json(path,plan)
    checkpoint()
    blocks=[['DECODE_ONLY','FULL','CP256','CP512'],
        ['DECODE_ONLY','CP1024','LP_AUTO','LP_G4'],
        ['DECODE_ONLY','LP_G8','LP_G16','LP_G24'],['DECODE_ONLY','LP_G48']]
    for i,arms in enumerate(blocks):
        run=f.launch(f'discovery_block{i}_{v}',arms,3,timeout=1000)
        plan['discovery_runs'].append(str(run));checkpoint()
    rows=sum((clean_rows(Path(run)) for run in plan['discovery_runs']),[])
    # AUTO and G4 are the same pinned algorithm; pool their observations,
    # rather than selecting the noisier minimum as two independent choices.
    candidates=[dict(r,arm='LP_AUTO' if r['arm']=='LP_G4' else r['arm'])
                for r in rows if r['modality']=='VISUAL_PREFILL' and r['arm'].startswith('LP_')]
    summaries=summarize(candidates)
    eligible=[r for r in summaries if r['peak_itl_ratio']['mean']<=1.5]
    best=min(eligible,key=lambda r:r['ttft_ms']['mean']) if eligible else min(summaries,key=lambda r:r['peak_vsel_ms']['mean'])
    selection={'best_arm':best['arm'],'best_groups':groups_for(best['arm'],2029),
        'slo_eligible':bool(eligible),'selection_contract':str(ROOT/'artifacts/manifests/analysis_contract_v4.json' if v=='v6' else ROOT/'artifacts/manifests/analysis_contract_v3.json'),
        'selection_runs':plan['discovery_runs'],'discovery_summary':summarize(rows),
        'alias_pooled_selection_summary':summaries,'selection_scope':'DISCOVERY_ONLY_BEFORE_FINAL'}
    atomic_json(ROOT/'artifacts/manifests'/f'frozen_final_selection_{v}.json',selection)
    primary=list(dict.fromkeys(['FULL','CP512','LP_AUTO',best['arm']]))
    plan['primary']=primary;plan['frozen_selection']=selection;checkpoint()
    # <=48 physical waves/session, including the complete warmup block.
    per_block=48//(len(primary)*4)-1
    offset=0;block=0
    while offset<10:
        count=min(per_block,10-offset)
        run=f.launch(f'final_block{block}_{v}',primary,count,paired=True,decode=128,
                     repeat_offset=offset,timeout=1500)
        plan['final_runs'].append(str(run));checkpoint();offset+=count;block+=1
    for i,arm in enumerate(primary):
        run=f.launch(f'component_block{i}_{v}',[arm],3,trace=True,light=True,
                     paired=True,overhead=True,decode=128,timeout=1000)
        plan['component_runs'].append(str(run));checkpoint()
    extra=[x for x in ('CP256','CP1024','LP_G8','LP_G16','LP_G24','LP_G48') if x not in primary]
    for i in range(0,len(extra),3):
        run=f.launch(f'extra_components_block{i//3}_{v}',extra[i:i+3],3,
                     trace=True,light=True,decode=128,timeout=1000)
        plan['extra_component_runs'].append(str(run));checkpoint()
    # Same-session Full reference for exact source-router hashes in every block.
    for i,arms in enumerate((['FULL','CP256','CP512'],['FULL','CP1024','LP_AUTO'],
                            ['FULL','LP_G8','LP_G16'],['FULL','LP_G24','LP_G48'])):
        run=f.launch(f'route_block{i}_{v}',arms,3,trace=True,timeout=1000)
        plan['route_runs'].append(str(run));checkpoint()
    plan['status']='MEASUREMENTS_COMPLETE';plan['reporting']='PENDING';checkpoint()
    print('MEASUREMENTS_COMPLETE_BLOCKED_'+v.upper(),flush=True)


if __name__=='__main__':main()

"""One bounded continuation after diagnosed startup/trace-only repairs.

    v6 clean/discovery evidence stays immutable. Failed startup is excluded.
    No algorithm reselection, automatic retries, or latency-model substitution.
"""
import hashlib
import os
from pathlib import Path

from poc_native_lp_ep4.contract import atomic_json, read_json
from poc_native_lp_ep4.scripts import finish_campaign as f


def main():
    root=Path('poc_native_lp_ep4');old=read_json(root/'artifacts/campaign_blocks_v6.json')
    assert len(old['discovery_runs'])==4 and len(old['final_runs'])==5
    for saved in old['discovery_runs']+old['final_runs']:
        assert read_json(Path(saved)/'status.json')['ok']
    failed=root/'artifacts/runs/component_block3_v6'
    assert not read_json(failed/'status.json')['ok']
    assert 'EADDRINUSE' in (root/'artifacts/component_block3_v6.log').read_text()
    assert not any((failed/f'worker.dp{rank}.json').exists() for rank in (0,1)), 'startup failure must precede measured waves'
    reference=root/'artifacts/runs/release_correctness_v4/source_signature.json'
    hashes=read_json(reference)
    for source in ('poc_native_lp_ep4/native.py','poc_native_lp_ep4/contract.py'):
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==hashes[source]
    f.VERSION='v7';os.environ['NATIVE_LP_EVENT_POOL']='1'
    best=old['frozen_selection']['best_arm']
    plan={**old,'version':'v7','status':'RUNNING','component_runs':[],
        'extra_component_runs':[],'route_runs':[],
        'legacy_component_runs':old['component_runs'],
        'failed_runs':[{'path':str(failed),'reason':'EADDRINUSE 51694 before MODEL_READY; no measurement'}],
        'source_plan':'poc_native_lp_ep4/artifacts/campaign_blocks_v6.json',
        'original_code_checkpoint':'6e318988',
        'repairs':['non-ephemeral consecutive rendezvous ports',
                   'reusable CUDA events, warmed before injection and reused after drain'],
        'clean_lane_repeated':False,'best_lp_reselected':False,
        'native_model_math_changed':False}
    destination=root/'artifacts/campaign_blocks_v7.json'
    if destination.exists():raise FileExistsError(destination)
    def checkpoint():atomic_json(destination,plan)
    checkpoint()
    gate=f.launch('pooled_correctness_v7',['FULL','LP_AUTO',best],3,
        trace=True,light=True,correctness=True,timeout=1100)
    f.check_correctness(gate,'pooled_validation_v7.json')
    plan['correctness_runs']=old['correctness_runs']+[str(gate)];checkpoint()
    # <=48 waves per engine; whole warmup and paired bypass/trace retained.
    for i,arms in enumerate((['FULL','CP512'],['LP_AUTO',best],
                           ['CP256','CP1024','LP_G16'],['LP_G24','LP_G48'])):
        run=f.launch(f'pooled_components_block{i}_v7',arms,3,trace=True,light=True,
            paired=i<2,overhead=True,decode=128,timeout=1500)
        plan['component_runs'].append(str(run));checkpoint()
    for i,arms in enumerate((['FULL','CP256','CP512'],['FULL','CP1024','LP_AUTO'],
                           ['FULL','LP_G8','LP_G16'],['FULL','LP_G24','LP_G48'])):
        run=f.launch(f'route_block{i}_v7',arms,3,trace=True,decode=128,timeout=1200)
        plan['route_runs'].append(str(run));checkpoint()
    plan['status']='MEASUREMENTS_COMPLETE';plan['reporting']='PENDING';checkpoint()
    print('MEASUREMENTS_COMPLETE_V7_CONTINUATION',flush=True)


if __name__=='__main__':main()

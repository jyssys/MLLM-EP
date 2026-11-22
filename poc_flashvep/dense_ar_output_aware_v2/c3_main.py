"""Freeze strong fixed configurations and run paired continuous-serving traces."""
import argparse
import hashlib
import json
import time
from .common import ROOT,write,digest,dispatch_allowed
from .c3_campaign import qualified_registry,performance,launch,wait_file
from .c3_policy import policy_registry,screen_score,FAMILIES

def select():
    plan=json.loads((ROOT/'jobs/c3_policy_discovery_v2.json').read_text());registry=policy_registry();chosen={};scores=[]
    for axis in FAMILIES:
        candidates=[]
        for label,item in registry.items():
            if item['family']!=axis:continue
            common=f'NativeCommon_Cap{item["cap"]}_Credit0'
            if common not in registry:continue
            names=[b['name'] for b in plan['blocks'] if b['label']==label]
            if not names or not all((ROOT/'analysis'/(name+'_metrics.json')).exists() for name in names):continue
            score=screen_score(label,plan['blocks'],common);scores.append(dict(score,family=axis,common_reference=common))
            if score['quality']:candidates.append(score)
        assert candidates,('No qualified candidate for final family',axis)
        best=min(candidates,key=lambda r:(not r['discovery_point_decode_screen'],r['mean_Native_normalized_TTFT'],-r['mean_goodput'],r['label']))
        item=registry[best['label']];chosen[axis]=dict(item=item,score=best,
            common=f'NativeCommon_Cap{item["cap"]}_Credit0',safety='POINT_SCREEN_ONLY' if best['discovery_point_decode_screen'] else 'SAFETY_UNCONFIRMED')
    result=dict(time=time.time(),chosen=chosen,scores=scores,heldout_performance_seen=False,
        scope='Configurations are frozen before Main400 and heldout192 performance. '
            'Stock Token retains the original compiled full-model path. Its cut,credit and cap are tuned,not weakened to Mode0. '
            'Discovery point safety does not establish the required independent-start CI gate. '
            'An unsafe fastest retained family is reported but cannot support a serving-success claim.')
    path=ROOT/'configs/c3_final_static_selection_v2.json';assert not path.exists();write(path,result)
    print('STRONG_FIXED_CONFIGURATIONS_FROZEN',{k:v['item']['label'] for k,v in chosen.items()},flush=True)
    return result

def plan(selection,heldout=False,replay=False):
    grid=json.loads((ROOT/'configs/c3_common_qps_grid_v2.json').read_text());registry=policy_registry();original=qualified_registry()
    qpss=grid['QPS_grid'];knee=grid['reference_knee']
    keys=sorted({min(qpss,key=lambda q:abs(q-knee)),min(qpss,key=lambda q:abs(q-1.2*knee))})
    if heldout or replay:qpss=keys
    dataset='balanced400_v2_heldout.json' if heldout else 'longbench_balanced400_v2.json'
    mode='Heldout192' if heldout else 'Main400';mode+='Replay' if replay else 'Natural'
    rows=json.loads((ROOT/'data'/dataset).read_text());lengths=None
    if replay:
        state=json.loads((ROOT/'state/official_compiled_c1_reference_complete_v2.json').read_text())
        refs={r['prompt_id']:r['output_tokens'] for r in map(json.loads,(ROOT/'raw'/state['name']/'requests.jsonl').read_text().splitlines())}
        lengths=[refs[r['prompt_id']] for r in rows];assert all(1<=x<=2048 for x in lengths)
    lanes=[[],[],[],[]];blocks=[]
    # Important points run first. This order is frozen before Main results.
    order=keys+[q for q in qpss if q not in keys]
    for qps in order:
        lane=grid['QPS_grid'].index(qps)%2;pair=[2*lane,2*lane+1]
        nstarts=3 if heldout or qps in keys else 1
        for start in range(nstarts):
            seed=(2026112281 if heldout else 2026112271)+start
            actions={axis:value['item'] for axis,value in selection['chosen'].items()}
            actions.update(NativeStrong=original['NativeStrong'],Agg=original['Agg']);references={}
            for axis,value in selection['chosen'].items():
                label='NativeCommon'+str(value['item']['cap']);actions[label]=registry[value['common']];references[axis]=label
            references['NativeStrong']='NativeStrong';references['Agg']='NativeStrong'
            for label in list(actions):
                if label.startswith('NativeCommon'):references[label]=label
            labels=sorted(actions,key=lambda label:hashlib.sha256(f'{mode}:{qps}:{start}:{label}'.encode()).hexdigest())
            arms={}
            for label in labels:
                name=f'V2_C3_{mode}_q{qps}_s{start}_{label}'
                performance(actions[label],name,dataset,pair,qps,seed,replay_lengths=lengths)
                lanes[lane].append(name);arms[label]=name
            blocks.append(dict(split=mode,qps=qps,start=start,seed=seed,pair=pair,dataset=dataset,arms=arms,NativeCommon_for=references))
    filename=('c3_heldout_replay_v2.json' if heldout else 'c3_main_replay_v2.json') if replay else 'c3_heldout_serving_v2.json' if heldout else 'c3_main_serving_v2.json'
    path=ROOT/'jobs'/filename;assert not path.exists()
    write(path,dict(time=time.time(),lanes=lanes,blocks=blocks,stop_on_failure=False,important_QPS=keys,
        dataset_SHA256=digest(ROOT/'data'/dataset),selection_SHA256=digest(ROOT/'configs/c3_final_static_selection_v2.json'),
        scope='Natural EOS Main400 includes both discovery and heldout requests. Heldout192 excludes C3 tuning contexts. '
            'All methods within a paired block use the same pair,original requests,order,arrival offsets and output contract. '
            'Independent starts initialize fresh actor processes. Compiler caches may persist. '
            'Common-policy Native exactly matches each selected D cap and decode-first/READ/registration deferral. '
            'NativeStrong remains unchanged. Important QPS are fixed from the capacity pilot,not selected for an axis win. '
            'Replay,when requested,is an explicitly separate fixed-output diagnostic with independent compiled C1 lengths.'))
    return filename

def replay_needed(heldout=True):
    stem='c3_heldout_serving_v2' if heldout else 'c3_main_serving_v2'
    path=ROOT/'analysis'/f'{stem}_analysis.json'
    if not path.exists():return False
    data=json.loads(path.read_text());flagged=[r for r in data['comparisons'] if r['output_work_unmatched']]
    write(ROOT/'state'/f'{stem}_natural_output_work_replay_trigger_v2.json',dict(time=time.time(),required=bool(flagged),flagged=flagged,
        threshold_fraction=.05,scope='Prospectively frozen diagnostic trigger. Do not replace natural-EOS results or use EOS lengths online.'))
    return bool(flagged)

def analyze(name):
    from .c3_analysis import analyze,figures
    result=analyze(name);figures(result,name.removesuffix('.json'))

def run():
    wait_file(ROOT/'state/c3_policy_tuning_complete_v2.json');selection=select()
    name=plan(selection);launch(name);analyze(name)
    name=plan(selection,heldout=True);launch(name);analyze(name)
    if replay_needed(heldout=False) and dispatch_allowed():
        name=plan(selection,replay=True);launch(name);analyze(name)
    if replay_needed() and dispatch_allowed():
        name=plan(selection,heldout=True,replay=True);launch(name)
        # Task quality is inherited,not re-scored after forced post-EOS generation.
        analyze(name)
        write(ROOT/'state/c3_heldout_replay_complete_v2.json',dict(time=time.time(),plan=name,complete=True,evidence='DIAGNOSTIC'))
    write(ROOT/'state/c3_main_and_heldout_complete_v2.json',dict(time=time.time(),complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','select'));args=p.parse_args();result=globals()[args.action]()
    if result is not None:print(result,flush=True)

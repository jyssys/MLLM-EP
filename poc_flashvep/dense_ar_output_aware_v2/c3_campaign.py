"""Stage clean pilots and prospective C3 geometry discovery."""
import argparse
import json
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write,digest,dispatch_allowed,initialize
from .runner import save,signature

def get(name):return json.loads((ROOT/'configs'/(name+'.json')).read_text())

def wait_file(path):
    while not path.exists():
        assert dispatch_allowed(),'Dispatch budget ended before this phase'
        time.sleep(5)
    return json.loads(path.read_text())

def qualified_registry():
    from .work_contract import validate
    plan=json.loads((ROOT/'configs/c3_family_qualification_v4_schedule_fix1.json').read_text());items=[]
    for item in plan['items']:
        proof_path=ROOT/'state'/item['gate']
        if proof_path.exists() and json.loads(proof_path.read_text())['passed']:
            cfg=get(item['jobs'][-1])
            try:validate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text()))
            except AssertionError:continue
            items.append(item)
    repaired=json.loads((ROOT/'configs/c3_agg_observer_repair_v5.json').read_text())['items'][0]
    gate=ROOT/'state'/repaired['gate']
    if gate.exists() and json.loads(gate.read_text())['passed']:
        items=[r for r in items if r['label']!='Agg']+[repaired]
    for filename in ('c3_stock_geometry_repair_v6.json','c3_fractional_stock_frontier_v7.json'):
        fixed=ROOT/'configs'/filename
        if not fixed.exists():continue
        for item in json.loads(fixed.read_text())['items']:
            gate=ROOT/'state'/item['gate']
            if gate.exists() and json.loads(gate.read_text())['passed']:
                items=[r for r in items if r['label']!=item['label']]+[item]
    return {r['label']:r for r in items}

def performance(item,name,dataset,pair,qps,seed,output=None,replay_lengths=None,suffix_override=None,observation=None):
    from .work_contract import regenerate,validate
    cfg=get(item['jobs'][-1]);rows=json.loads((ROOT/'data'/dataset).read_text());n=len(rows)
    cfg.update(name=name,dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),count=n,gpus=list(pair),qps=qps,seed=seed,
        arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(1/qps,n-1))].tolist(),
        correctness_only=False,axis_state_capture=False,state_capture=False,gpu_state_digest=False,
        numerical_probe=False,numerical_probe_sync=False,kernel_activity=False,serial=False,qualification_gate_v2=item['gate'])
    if output is not None:cfg.update(output_cap=output,ignore_eos=True)
    if replay_lengths is not None:cfg.update(diagnostic_controlled_output_replay=True,replay_output_lengths=replay_lengths,ignore_eos=True)
    regenerate(cfg,rows);validate(cfg,rows)
    if suffix_override is not None:
        cfg['suffix_by_request']=suffix_override;validate(cfg,rows)
    if observation:cfg.update(observation)
    assert signature(cfg)==json.loads((ROOT/'state'/item['gate']).read_text())['signature']
    save(cfg);return name

def launch(plan_name):
    processes=[];plan=json.loads((ROOT/'jobs'/plan_name).read_text())
    for lane,jobs in enumerate(plan['lanes']):
        if not jobs:continue
        log=ROOT/'logs'/f'{plan_name.removesuffix(".json")}_lane{lane}.log'
        assert not log.exists(),'Keep prior logs; resume with a new plan version'
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_output_aware_v2.runner',plan_name,str(lane)],
                cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(child)
    while any(p.poll() is None for p in processes):
        write(ROOT/'state/c3_campaign_active_children_v2.json',dict(time=time.time(),plan=plan_name,
            children=[dict(pid=p.pid,returncode=p.poll()) for p in processes]));time.sleep(5)
    assert all(p.returncode==0 for p in processes),[(p.pid,p.returncode) for p in processes]
    for lane,jobs in enumerate(plan['lanes']):
        if not jobs:continue
        receipt=json.loads((ROOT/'state'/f'{plan_name.removesuffix(".json")}_lane{lane}_complete.json').read_text())
        assert receipt['complete'] and all(r['status']=='COMPLETE' for r in receipt['results']),receipt

def pilots():
    contract=json.loads((ROOT/'configs/c3_clean_performance_contract_v2.json').read_text());registry=qualified_registry()
    assert all(k in registry for k in ('NativeStrong','Agg'))
    lanes=[[],[],[],[]];cells=[]
    for index,qps in enumerate(contract['pilot_QPS']):
        lane=index%2;pair=[2*lane,2*lane+1]
        for label in ('NativeStrong','Agg'):
            name=f'V2_C3Capacity_q{qps}_{label}_Clean64'
            performance(registry[label],name,contract['pilot_dataset'],pair,qps,2026112261)
            lanes[lane].append(name);cells.append(dict(label=label,qps=qps,name=name,pair=pair))
    path=ROOT/'jobs/c3_clean_capacity_pilots_v2.json'
    if not path.exists():write(path,dict(time=time.time(),lanes=lanes,cells=cells,stop_on_failure=True,
        contract_SHA256=digest(ROOT/'configs/c3_clean_performance_contract_v2.json')))
    return path.name

def freeze_grid():
    pilot=json.loads((ROOT/'jobs/c3_clean_capacity_pilots_v2.json').read_text());knees={};measured=[]
    for label in ('NativeStrong','Agg'):
        values=[]
        for cell in sorted((r for r in pilot['cells'] if r['label']==label),key=lambda r:r['qps']):
            metric=json.loads((ROOT/'analysis'/(cell['name']+'_metrics.json')).read_text())
            span=metric['last_arrival_seconds'];drain=metric['measurement_seconds']-span
            values.append(dict(cell,TTFT_p95_ms=metric['TTFT_p95_ms'],achieved_QPS=metric['completed_requests_per_second'],
                actual_arrival_QPS=metric['actual_arrival_QPS'],
                drain_fraction=drain/max(span,1e-9),evidence='MEASURED_ONLINE',sustainable_capacity=False))
        first=values[0]['TTFT_p95_ms'];hits=[r for r in values if r['achieved_QPS']<.9*r['actual_arrival_QPS'] or
            (r['TTFT_p95_ms']>1.5*first and r['drain_fraction']>.25)]
        knees[label]=dict(qps=hits[0]['qps'] if hits else values[-1]['qps'],observed_knee=bool(hits))
        measured+=values
    knee=min(r['qps'] for r in knees.values());grid=sorted({max(.2,round(knee*v,2)) for v in (.2,.35,.5,.65,.8,1.,1.2,1.5)})
    record=dict(time=time.time(),knees=knees,reference_knee=knee,QPS_grid=grid,pilot=measured,
        knee_amendment_SHA256=digest(ROOT/'configs/c3_capacity_knee_amendment_v2.json'),
        discovery_QPS=sorted({min(grid,key=lambda x:abs(x-.7*knee)),min(grid,key=lambda x:abs(x-1.2*knee))}),
        scope='Prospectively specified finite-cohort knee heuristic. The grid is frozen before candidate performance. '
            'Different pilot pairs are recorded. These results do not establish sustainable capacity.')
    target=ROOT/'configs/c3_common_qps_grid_v2.json';assert not target.exists();write(target,record)
    print('C3_COMMON_GRID_FROZEN',record,flush=True);return record

def discovery():
    grid=json.loads((ROOT/'configs/c3_common_qps_grid_v2.json').read_text());registry=qualified_registry();lanes=[[],[],[],[]];blocks=[]
    for lane,qps in enumerate(grid['discovery_QPS']):
        pair=[2*lane,2*lane+1];order=['NativeStrong','NativeCommon2','Agg']+sorted(k for k in registry if k not in ('NativeStrong','NativeCommon2','Agg'))
        for label in order:
            if label not in registry:continue
            name=f'V2_C3Geometry_q{qps}_{label}_Discovery208'
            performance(registry[label],name,'balanced400_v2_discovery.json',pair,qps,2026112262)
            lanes[lane].append(name);blocks.append(dict(qps=qps,label=label,name=name,pair=pair,gate=registry[label]['gate']))
    path=ROOT/'jobs/c3_clean_geometry_discovery_v2.json';assert not path.exists()
    write(path,dict(time=time.time(),lanes=lanes,blocks=blocks,stop_on_failure=False,
        contract_SHA256=digest(ROOT/'configs/c3_clean_performance_contract_v2.json'),
        scope='Each load block uses one identical pair,arrival trace and discovery208 request order. '
            'Two loads can use different equal-SKU pairs. Do not pool unmatched load/pair measurements as independent starts.'))
    return path.name

def run():
    initialize()
    for lane in (0,1):wait_file(ROOT/'state'/f'c3_family_lane{lane}_complete_v4.json')
    wait_file(ROOT/'state/c3_agg_observer_repair_v5_lane0_complete_v4.json')
    for lane in (0,1):wait_file(ROOT/'state'/f'c3_stock_geometry_repair_v6_lane{lane}_complete_v4.json')
    for lane in (0,1):wait_file(ROOT/'state'/f'c3_fractional_stock_frontier_v7_lane{lane}_complete_v4.json')
    launch(pilots());freeze_grid();launch(discovery())
    write(ROOT/'state/c3_pilot_and_geometry_complete_v2.json',dict(time=time.time(),complete=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('run','pilots','freeze_grid','discovery'));args=parser.parse_args()
    result=globals()[args.action]()
    if result is not None:print(result,flush=True)

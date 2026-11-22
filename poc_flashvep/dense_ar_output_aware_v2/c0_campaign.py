"""Run the full predeclared output grid with separate discovery and confirmation."""
import argparse
import fcntl
import json
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write,digest,initialize,dispatch_allowed
from .runner import save,signature

OUTPUTS=(8,16,64,128,192)
CUTS=(2,4,8,12)

def config(proto,name,dataset,pair,seed,qps,output,serial=False):
    cfg=json.loads((ROOT/'configs'/(proto+'.json')).read_text());rows=json.loads((ROOT/'data'/dataset).read_text());n=len(rows)
    cfg.update(name=name,dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),count=n,gpus=list(pair),qps=qps,seed=seed,
        arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(1/qps,n-1))].tolist(),
        output_cap=output,ignore_eos=True,serial=serial,correctness_only=False,axis_state_capture=False,state_capture=False,
        gpu_state_digest=False,numerical_probe=False,numerical_probe_sync=False)
    return cfg

def pilots():
    initialize();path=ROOT/'jobs/c0_native_service_pilots_v2.json'
    if path.exists():return json.loads(path.read_text())
    lanes=[[] for _ in range(4)];cells=[]
    for bucket in range(4):
        for output in OUTPUTS:
            name=f'V2_C0Capacity_I{bucket}_O{output}_Serial16'
            cfg=config('V2_C0_Native_State8',name,f'input{bucket}_discovery_exact16.json',(2*bucket,2*bucket+1),2026112221,20.,output,True)
            cfg['qualification_gate_v2']='c0_Native_gate_v2.json';save(cfg);lanes[bucket].append(name)
            cells.append(dict(input_bucket=bucket,output=output,name=name))
    plan=dict(time=time.time(),lanes=lanes,cells=cells,stop_on_failure=True,
        rule='For each input bin set one common absolute QPS across all five outputs:0.85 times the minimum native two-stage host-demand rate. '
            'The rate uses1/max(mean P-submit-to-ready,mean D-submit-to-finish). It is a pilot estimate,not measured sustainable capacity.',
        outputs=OUTPUTS,cuts=CUTS,source_capsule_SHA256=json.loads((ROOT/'configs/V2_C0_Native_State8.json').read_text())['source_capsule_SHA256'])
    write(path,plan);return plan

def matrix():
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.service_accounting import analyze
    from ..dense_ar_longbench400_pd.geometry import matched_suffixes
    path=ROOT/'jobs/c0_output_matrix_v2.json';assert not path.exists()
    pilot=json.loads((ROOT/'jobs/c0_native_service_pilots_v2.json').read_text());rates={i:[] for i in range(4)};demands=[]
    for cell in pilot['cells']:
        a=analyze(cell['name']);p=a['means']['P_submit_to_ready_ms'];d=a['means']['D_submit_to_finish_ms'];rate=1000/max(p,d)
        rates[cell['input_bucket']].append(rate);demands.append(dict(cell,P_host_ms=p,D_host_ms=d,host_demand_rate=rate))
    qps={i:round(.85*min(v),6) for i,v in rates.items()}
    lanes=[[] for _ in range(4)];blocks=[]
    for split,seeds in (('discovery',[2026112231]),('heldout',[2026112241,2026112242,2026112243])):
        for bucket in range(4):
            data=f'input{bucket}_{split}_exact16.json';rows=json.loads((ROOT/'data'/data).read_text())
            lane=2+bucket%2;pair=[2*lane,2*lane+1]
            for output in OUTPUTS:
                for start,seed in enumerate(seeds):
                    native=f'V2_C0_{split}_I{bucket}_O{output}_s{start}_Native'
                    cfg=config('V2_C0_Native_State8',native,data,pair,seed,qps[bucket],output)
                    cfg['qualification_gate_v2']='c0_Native_gate_v2.json';save(cfg);lanes[lane].append(native)
                    for cut in CUTS:
                        suffix,work=matched_suffixes([r['input_tokens'] for r in rows],cut);assert work['exact_match']
                        arms={}
                        order=('Layer','Token') if (seed+cut+output)%2 else ('Token','Layer')
                        for axis in order:
                            name=f'V2_C0_{split}_I{bucket}_O{output}_C{cut}_s{start}_{axis}'
                            cfg=config(f'V2_C0_{axis}{cut}_State8',name,data,pair,seed,qps[bucket],output)
                            cfg['qualification_gate_v2']=f'c0_{axis}{cut}_gate_v2.json'
                            if axis=='Token':cfg.update(suffix_by_request={f'q{i}':s for i,s in enumerate(suffix)},aggregate_work_audit=work)
                            save(cfg);lanes[lane].append(name);arms[axis]=name
                        blocks.append(dict(split=split,input_bucket=bucket,output=output,cut=cut,start=start,seed=seed,qps=qps[bucket],
                            pair=pair,dataset=data,Native=native,**arms))
    for jobs in lanes:
        for name in jobs:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());proof=json.loads((ROOT/'state'/cfg['qualification_gate_v2']).read_text())
            assert proof['passed'] and signature(cfg)==proof['signature'],('Composition mismatch before dispatch',name)
    plan=dict(time=time.time(),lanes=lanes,blocks=blocks,QPS_by_input=qps,pilot_demands=demands,stop_on_failure=True,
        source_capsule_SHA256=pilot['source_capsule_SHA256'],point_rule='Ratio of paired independent-start TTFT p95 means; do not recolor a point when CI crosses1.',
        evidence_scope='C0 actual1/1/1,credit4,KV42GiB/block128,B8192/S1024. Mode0 token is a mechanism control,not the strong C3 stock-token baseline.',
        confirmation='All80 cells have3 new paired server-start/arrival-seed blocks on context-disjoint original held-out requests. Discovery has one start per cell. '
            'QPS stays constant across outputs within an input bin. This changes capacity-relative load when output changes. '
            'C3 heldout is held out from C3 tuning,not globally unseen after C1/C0 exposure.',
        allocation='Freeze before any v2 axis timings. Keep C0 on4/5 and6/7;reserve0/1 and2/3 for the primary C3 implementation,qualification and serving workstreams. '
            'Within every method comparison,the actual GPU pair stays identical. Native pilot capacity is a same-SKU estimate,not cross-pair performance evidence.')
    write(path,plan);print('C0_FULL_OUTPUT_MATRIX_FROZEN',qps,sum(map(len,lanes)),flush=True)
    return plan

def launch(plan_name):
    children=[]
    for lane in range(4):
        handle=(ROOT/'logs'/f'{plan_name.removesuffix(".json")}_lane{lane}.log').open('w')
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_output_aware_v2.runner',plan_name,str(lane)],
            cwd=REPO,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        handle.close();children.append(child)
    return children

def wait(children,stage):
    while any(p.poll() is None for p in children):
        write(ROOT/'state/c0_campaign_controller.json',dict(time=time.time(),stage=stage,
            children=[dict(pid=p.pid,returncode=p.poll()) for p in children]))
        time.sleep(5)
    assert all(p.returncode==0 for p in children),[(p.pid,p.returncode) for p in children]

def run():
    initialize();pilots()
    wait(launch('c0_native_service_pilots_v2.json'),'NATIVE_SERVICE_PILOTS')
    matrix();wait(launch('c0_output_matrix_v2.json'),'OUTPUT_AWARE_FULL_MATRIX')
    write(ROOT/'state/c0_campaign_complete_v2.json',dict(time=time.time(),complete=True))

def resume_pilots():
    initialize()
    for lane in range(4):
        path=ROOT/'state'/f'c0_native_service_pilots_v2_lane{lane}_complete.json'
        while not path.exists():assert dispatch_allowed();time.sleep(5)
        receipt=json.loads(path.read_text());assert receipt['complete'] and all(r['status']=='COMPLETE' for r in receipt['results']),receipt
    matrix();wait(launch('c0_output_matrix_v2.json'),'OUTPUT_AWARE_FULL_MATRIX_TWO_PAIRS')
    write(ROOT/'state/c0_campaign_complete_v2.json',dict(time=time.time(),complete=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('pilots','matrix','run','resume_pilots'));args=parser.parse_args();globals()[args.action]()

"""Build held-out rate cells from actual finite service measurements."""
import argparse
import statistics
from .common import *
from .summary_tables import records,percentile
from .phase_b_configs import pd_config
from .prepare import GRID

def capacity(model,cell,chunk=256,pool=None):
    groups={}
    for folder in (ROOT/'raw').iterdir():
        if not folder.name.startswith('phase_a_'+model):continue
        if 'event_overhead' in folder.name:continue
        status=folder/'status.json'
        if not status.exists() or json.loads(status.read_text())['status']!='COMPLETE':continue
        for row in records(folder/'cohorts.jsonl'):
            if row.get('status')!='COMPLETE' or row.get('chunk')!=chunk or row['job'].get('cell')!=cell:continue
            if row['job']['cap']!=GRID[cell][1]:continue
            if pool is not None and row['KV_blocks']!=pool:continue
            key=row['batch'];groups.setdefault(key,[]).append(row)
    if not groups:raise ValueError('No measured independent capacity for '+model+'/'+cell)
    values=[]
    for b,rows in groups.items():
        p=statistics.median(r['prefill_host_service_s'] for r in rows)
        d=statistics.median(r['decode_native_boundary_service_s'] for r in rows)
        values.append(dict(batch=b,mu_P=b/p,mu_D=b/d,P_service_s=p,D_service_s=d,repeats=len(rows)))
    independentP=[]
    for folder in (ROOT/'raw').glob(f'phase_a_{model}_ponly*'):
        if not (folder/'status.json').exists() or json.loads((folder/'status.json').read_text())['status']!='COMPLETE':continue
        for row in records(folder/'cohorts.jsonl'):
            if row.get('status')=='COMPLETE' and row['job'].get('cell')==cell and row.get('chunk')==chunk:
                independentP.append(row['batch']/row['prefill_host_service_s'])
    return dict(mu_P=max([r['mu_P'] for r in values]+independentP),mu_D=max(r['mu_D'] for r in values),samples=values,
        dedicated_P_only_service_samples=independentP,
        scope='Finite no-backpressure service rates. This is a rate-planning estimate, not measured sustainable serving capacity.')

def make_screen():
    names=[];planning=[]
    for model in ('ouro','huginn'):
        gate=ROOT/'state'/f'{model}_native_pd_gate.json'
        assert gate.exists() and json.loads(gate.read_text())['status']=='PASS_NATIVE_PD_TRANSPORT','Native PD correctness evidence is not complete'
        gate=ROOT/'state'/f'{model}_qualification32.json'
        assert gate.exists() and json.loads(gate.read_text())['status']=='PASS_32_NATIVE_PD_QUALIFICATION','The 32-request qualification is not complete'
        gate=ROOT/'state'/f'{model}_qualification2.json'
        assert gate.exists() and json.loads(gate.read_text())['status']=='PASS_2_NATIVE_PD_QUALIFICATION','The partial-tail long-prompt qualification is not complete'
        for cell in ('G1','G2','G3','G4','G5','G6','C1','C2'):
            measured=capacity(model,cell,pool=8192 if model=='ouro' else 18432)
            rate=.8*min(measured['mu_P'],measured['mu_D'])
            c=pd_config(model,f'phase_b_{model}_{cell.lower()}_screen_start0',cell,24,rate)
            c['capacity_planning']=measured;c['rate_factor']=.8
            path=ROOT/'configs'/(c['name']+'.json');assert not path.exists();write(path,c);names.append(path.name)
            planning.append(dict(model=model,cell=cell,rate=rate,**measured))
    write(ROOT/'configs/phase_b_screen_plan.json',dict(configs=names));write(ROOT/'configs/rate_planning.json',planning)

def make_calibration():
    names=[]
    for model in ('ouro','huginn'):
        cap=capacity(model,'G2',pool=8192 if model=='ouro' else 18432)
        # Eight disjoint, length-matched calibration prompts. Calibration is a finite low-rate cohort.
        rate=.25*min(cap['mu_P'],cap['mu_D'])
        c=pd_config(model,f'calibration_{model}_g2_start0','G2',8,rate,True)
        c['capacity_planning']=cap;c['calibration_serial']=True
        c['arrival_pattern']='CLOSED_LOOP_SERIAL_CALIBRATION_NOT_POISSON'
        path=ROOT/'configs'/(c['name']+'.json');assert not path.exists();write(path,c);names.append(path.name)
    write(ROOT/'configs/phase_b_calibration_plan.json',dict(configs=names))

def freeze_SLO():
    path=ROOT/'configs/frozen_SLO.json'
    assert not path.exists(),'Do not replace frozen SLO thresholds'
    result={}
    for model in ('ouro','huginn'):
        folder=ROOT/'raw'/f'calibration_{model}_g2_start0'
        assert json.loads((folder/'status.json').read_text())['status']=='COMPLETE'
        rows=[r for r in records(folder/'requests.jsonl') if not r['warmup']]
        assert len(rows)==8
        itls=[1000*(e['time']-e['previous_time'])/e['new_tokens'] for r in rows for e in r['token_events'] if not e['first']]
        result[model]=dict(TTFT_ms=2*percentile([r['ttft_ms'] for r in rows],.95),
            TPOT_ms=2*percentile([r['tpot_ms'] for r in rows],.95),ITL_ms=2*percentile(itls,.99),
            calibration_requests=8,calibration_run=folder.name,rule='2x calibration p95 TTFT / request TPOT, 2x population p99 ITL',
            time=time.time(),evidence='MEASURED_CALIBRATION_FROZEN_BEFORE_HELD_OUT')
    write(path,result)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['calibration','freeze_slo','screen']);a=p.parse_args()
    {'calibration':make_calibration,'freeze_slo':freeze_SLO,'screen':make_screen}[a.action]()

if __name__=='__main__':main()

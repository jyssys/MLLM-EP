"""Bounded secondary-length search after the original and primary controls."""
import copy
from .common import *
from .prepare import put
from .initial_campaign import run,gate
from .primary_campaign import freeze_slo,subset_parity
from .search_plan import make as primary_plan


def plan():
    destination=ROOT/'state/adaptive_length_plan.json'
    if destination.exists():return json.loads(destination.read_text())
    output=[]
    for model,length,cap in [('ouro',8192,32),('huginn',3072,32)]:
        primary=next(r for r in primary_plan()['primaries'] if r['model']==model)
        base=json.loads((ROOT/'configs'/(primary['gate'][0]+'.json')).read_text())
        base.update(length=length,cap=cap,state_capacity_tokens=length,
            token_budget=32768 if model=='ouro' else 2*length,timeout=900,pd_shutdown_timeout=300,
            gpus=[0,1] if model=='ouro' else [4,5],
            native_transport_limits=dict(base.get('native_transport_limits',{}),max_inflight_bytes=8*1024**3),
            workload=f'L{length}_O{cap}',P_only_profiler=False,nsys=False,
            observe_descriptor_cost=False,record_descriptor_replay=False)
        for key in ('capacity_reference','reference_capacity','rate_factor','rate_scope'):base.pop(key,None)
        item=dict(model=model,length=length,cap=cap,gate=[],calibration=[],capacity=[],pilots=[])
        for chunk in (length,2048):
            cfg=copy.deepcopy(base)
            cfg.update(name=f'secondary_{model}_l{length}_o{cap}_c{chunk}_gate8',chunk=chunk,
                phase='correctness_secondary_length',state_probe=True,verify_transfer=True)
            item['gate'].append(put(cfg,8))
            if model=='huginn':
                cfg=copy.deepcopy(base)
                cfg.update(name=f'calibration_{model}_l{length}_c{chunk}_serial16',chunk=chunk,
                    phase='calibration',state_probe=False,verify_transfer=False,calibration_serial=True)
                item['calibration'].append(put(cfg,16,split='calibration'))
            cfg=copy.deepcopy(base)
            cfg.update(name=f'capacity_secondary_{model}_l{length}_c{chunk}_p_only',chunk=chunk,
                worker='capacity_worker',capacity_mode='P_ONLY',phase='independent_capacity',
                state_probe=False,verify_transfer=False,gpus=[base['gpus'][0]],capacity=4,slots=4,
                jobs=[dict(batch=b,cap=1,repeats=3) for b in ((1,2,4) if model=='ouro' else (1,2,3))])
            item['capacity'].append(put(cfg,8))
            for rate in ([1.,1.5,2.,2.5] if model=='ouro' else [.6,.8,1.,1.2]):
                cfg=copy.deepcopy(base)
                cfg.update(name=f'secondary_{model}_l{length}_o{cap}_c{chunk}_q{str(rate).replace(".","p")}_pilot24',
                    chunk=chunk,phase='qps_pilot',state_probe=False,verify_transfer=False)
                item['pilots'].append(put(cfg,24,rate))
        output.append(item)
    result=dict(created=time.time(),items=output,
        scope='Ouro short-output control retains 8K. Huginn secondary input is 3072. All use supported 8-GiB wire credit and bounded pending transfers. New Huginn SLO freezes before its held-out pilots. Ouro retains the prior 8K Decode limits as an explicit output-sensitivity control.')
    write(destination,result)
    return result


def main():
    while True:
        path=ROOT/'state/primary_campaign.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        if state.get('stage')=='PRIMARY_MATRIX_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    ledger=[]
    for item in plan()['items']:
        valid=[]
        for name in item['gate']:
            write(ROOT/'state/adaptive_ladder.json',dict(stage='SECONDARY_CORRECTNESS',name=name,time=time.time()))
            okay=run(name) and gate(name,valid[0] if valid else None)
            ledger.append(dict(name=name,passed=okay))
            if not okay:break
            valid.append(name)
        if len(valid)!=2:continue
        for name in item['capacity']+item['calibration']:
            write(ROOT/'state/adaptive_ladder.json',dict(stage='SECONDARY_CAPACITY_CALIBRATION',name=name,time=time.time()))
            ledger.append(dict(name=name,passed=run(name)))
        if item['model']=='huginn' and not freeze_slo(item):continue
        reference=None
        for name in item['pilots']:
            write(ROOT/'state/adaptive_ladder.json',dict(stage='SECONDARY_QPS',name=name,time=time.time()))
            okay=run(name) and subset_parity(valid[0],name)
            if okay and reference:okay=subset_parity(reference,name)
            if reference is None:reference=name
            ledger.append(dict(name=name,passed=okay))
            write(ROOT/'state/adaptive_ladder_results.json',ledger)
            if not okay:break
    write(ROOT/'state/adaptive_ladder.json',dict(stage='ADAPTIVE_LADDER_COMPLETE_REVIEW_REQUIRED',time=time.time(),results=ledger))
    restore_if_idle()


if __name__=='__main__':main()

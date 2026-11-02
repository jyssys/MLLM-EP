"""Prepare the primary lengths after the original-length controls."""
import copy
from .common import *
from .prepare import put


def make():
    revised=ROOT/'state/primary_plan_v2.json'
    if revised.exists():return json.loads(revised.read_text())
    path=ROOT/'state/primary_plan.json'
    if path.exists():return json.loads(path.read_text())
    plan=[]
    sources=dict(ouro='recovery_ouro_8192_64_workspace128_pool25088_drain0_gate8_v2',
        huginn='recovery_huginn_3500_32_control_gate8')
    # These six coarse rates avoid a blind Cartesian sweep. Lower and optional
    # overload points remain declared but unrun unless the capacity evidence
    # makes them useful. Atomic and C2048 receive the same rate budget.
    rates=dict(ouro=[.5,1.,1.5,2.,2.5,3.],huginn=[.2,.4,.6,.8,1.,1.2])
    for model,length,cap in [('ouro',6144,64),('huginn',2560,32)]:
        base=json.loads((ROOT/'configs'/(sources[model]+'.json')).read_text())
        base.update(length=length,state_capacity_tokens=length,cap=cap,chunk=length,
            token_budget=32768 if model=='ouro' else 2*length,
            seed=2026110207,arrival_seed=2026110201,timeout=360,verify_transfer=False,state_probe=False,
            execution_control_family='ADMISSION_RECOVERY_PRIMARY',P_only_profiler=False,nsys=False,
            gpus=[2,3] if model=='ouro' else [6,7],independent_start=0)
        record=dict(model=model,length=length,cap=cap,gate=[],capacity=[],calibration=[],pilots=[])
        for chunk in (length,2048):
            cfg=copy.deepcopy(base)
            cfg.update(name=f'primary_{model}_l{length}_o{cap}_c{chunk}_gate8',chunk=chunk,
                phase='correctness',state_probe=True,verify_transfer=True)
            record['gate'].append(put(cfg,8))
            calibration=copy.deepcopy(base)
            calibration.update(name=f'calibration_{model}_l{length}_c{chunk}_serial16',chunk=chunk,
                phase='calibration',calibration_serial=True,timeout=600)
            record['calibration'].append(put(calibration,16,split='calibration'))
            local=copy.deepcopy(base)
            local.update(name=f'capacity_{model}_l{length}_c{chunk}_p_only',chunk=chunk,
                worker='capacity_worker',capacity_mode='P_ONLY',phase='independent_capacity',gpus=[base['gpus'][0]],
                capacity=4,slots=4,graph_rows=base.get('graph_rows',8),
                jobs=[dict(batch=b,cap=1,repeats=3) for b in (1,2,4)],timeout=600)
            record['capacity'].append(put(local,8))
            for rate in rates[model]:
                cfg=copy.deepcopy(base)
                cfg.update(name=f'primary_{model}_l{length}_o{cap}_c{chunk}_q{str(rate).replace(".","p")}_pilot24',
                    chunk=chunk,phase='qps_pilot')
                record['pilots'].append(put(cfg,24,rate))
        plan.append(record)
    # Additional native wire credit control. It follows the first four
    # original-length controls and changes only the supported credit limit.
    c=json.loads((ROOT/'configs'/(sources['ouro']+'.json')).read_text())
    c['native_transport_limits']=dict(c['native_transport_limits'],max_inflight_bytes=8*1024**3)
    c.update(name='recovery_ouro_8192_64_bounded_credit8g_gate8',phase='correctness')
    extra_gate=put(c,8)
    c=copy.deepcopy(c)
    c.update(name='recovery_ouro_8192_64_bounded_credit8g_pilot24',phase='qps_pilot',state_probe=False,verify_transfer=False)
    original=json.loads((ROOT/'configs/recovery_ouro_8192_64_original_pilot24_v2.json').read_text())
    extra_pilot=put(c,24,original['arrival_rate'])
    result=dict(created=time.time(),primaries=plan,extra_ouro_control=dict(gate=extra_gate,pilot=extra_pilot),
        deferred=dict(ouro=[4.,5.,7.,10.],huginn=[.05,.1,1.5,2.,3.,5.,10.]),
        priority='Original-length recovery, then new primary correctness and capacity, then two-chunk coarse QPS pilots. No Dynamic A.')
    write(path,result)
    return result


if __name__=='__main__':
    print(json.dumps(make(),indent=2))

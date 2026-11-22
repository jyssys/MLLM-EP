"""Compare legal compute deferral with original and device-quiescent paths."""
import argparse
import json
import time
from .common import ROOT,write,dispatch_allowed
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .run import execute
from .next_jobs import plan


def gate():
    checks=[]
    for variant in ('Layer','Token','Hybrid','StockToken'):
        path=ROOT/'state'/f'D_compute_throttle_{variant}_qualification_v1.json'
        if not path.exists():return False
        row=json.loads(path.read_text())
        if not row.get('passed') or not row.get('full400_passed'):return False
        checks.append(row)
    write(ROOT/'state/D_compute_throttle_gate_v1.json',dict(passed=True,full400_passed=True,
        qualified_axes=['Native','Layer','Token','Hybrid'],stock_token_model=True,
        checks=checks,evidence='CORRECTNESS_ONLY',
        scope='LegacyMode0 axes and default-compiled StockToken4 passed strict N1/N8 and Natural32/400 state/quality gates. '
            'Native uses the unchanged original model and normal one-query handoff. '
            'Compute deferral keeps remote-KV processing and continuous decode batching. No ITL or speed guarantee.'))
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('lane',type=int,choices=(0,1));args=parser.parse_args()
    capsule,sha=freeze('compute_throttle_AB_discovery_v1_20261121')
    qps,first=((1.8,2),(2.8,6))[args.lane]
    data=json.loads((ROOT/'data/balanced400_discovery.json').read_text())
    suffix,work=matched_suffixes([r['input_tokens'] for r in data],4)
    cells=[];jobs=[]
    actions=[('NativeStrong','Native','NONE',True),('NativeMatched','Native','NONE',False),
        ('NativeThrottle','Native','COMPUTE',False)]
    for axis in ('Layer','Token','Hybrid'):
        actions.extend((axis+label,axis,policy,False) for label,policy in
            (('Original','NONE'),('Quiescent','QUIESCENT'),('Throttle','COMPUTE')))
    actions.extend([('StockTokenOriginal','Token','NONE',True),('StockTokenThrottle','Token','COMPUTE',True)])
    for label,axis,policy,compiled in actions:
        extra={}
        if axis!='Native':extra['qualification_gate']='C3_continuous_actions_gate_v1.json'
        if axis=='Token':extra.update(suffix_by_request={'q'+str(i):v for i,v in enumerate(suffix)},aggregate_work_audit=work)
        if axis=='Hybrid':extra.update(hybrid_suffix_fraction=.125,partition_aligned_chunks=True)
        if axis=='Token' and compiled:extra.update(token_stock_model=True,qualification_gate='C3_stock_token_gate_v1.json')
        name=create(f'ThrottleAB_Discovery200_q{qps}_{label}_v1',axis,'balanced400_discovery.json',
            cut=0 if axis=='Native' else 4,pair=(first,first+1),qps=qps,seed=2026112291,
            execution_family='INSTALLED_DEFAULT_COMPILE' if compiled else 'MATCHED_NO_COMPILE',
            continuous_source_credit_limit=0 if compiled else 4,
            d_quiescent_admission=policy=='QUIESCENT',d_compute_throttle=policy=='COMPUTE',
            source_capsule=capsule,source_capsule_SHA256=sha,discovery_only=True,
            scope='Same-source,request and arrival A/B. Change D policy only within each same-compiler axis family. '
                'Quiescent admission blocks transfers and compute during active decode. Compute deferral still permits remote-KV work. '
                'Record actual batches. Keep the strong unmodified compiled Native and StockToken baselines. '
                'One discovery start is not a serving success or paired-start safety CI.',**extra)
        jobs.append(name);cells.append(dict(name=name,axis=axis,policy=policy,default_compiled=compiled))
    manifest=f'compute_throttle_AB_lane{args.lane}_v1';plan(manifest,jobs)
    write(ROOT/'configs'/(manifest+'.json'),dict(cells=cells,QPS=qps,source_capsule_SHA256=sha,
        evidence='FROZEN_CAUSAL_DISCOVERY_DESIGN',
        scope='Keep all original,serial-admission and compute-deferral outcomes. '
            'The policy can worsen TTFT or throughput. Never change SLOs or token-delivery timestamps.'))
    print('WAIT_COMPUTE_THROTTLE_CORRECTNESS',manifest,flush=True)
    while not gate():
        assert dispatch_allowed();time.sleep(5)
    for name in jobs:assert execute(name),name
    print('COMPUTE_THROTTLE_AB_COMPLETE',manifest,flush=True)


if __name__=='__main__':main()

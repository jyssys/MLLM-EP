"""Dispatch causal timings only after every admission-correctness stage passes."""
import argparse
import json
import time
from .common import ROOT,write,dispatch_allowed
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .next_jobs import plan
from .run import execute


def gate():
    checks=[]
    for axis in ('Layer','Token','Hybrid'):
        path=ROOT/'state'/f'D_quiescent_{axis}_qualification_v1.json'
        if not path.exists():return False
        value=json.loads(path.read_text())
        if not value.get('passed') or not value.get('full400_passed'):return False
        checks.append(value)
    write(ROOT/'state/D_quiescent_admission_gate_v1.json',dict(passed=True,full400_passed=True,
        qualified_axes=['Native','Layer','Token','Hybrid'],checks=checks,evidence='CORRECTNESS_ONLY',
        scope='The admission-only intervention passed its N1,N8,N32 and full400 state/lifetime/quality checks. '
            'Native keeps the original unmodified model. This gate does not prove decode safety or serving speed.'))
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('lane',type=int,choices=(0,1));args=parser.parse_args()
    capsule,sha=freeze('quiescent_admission_AB_discovery_v1_20261121')
    qps,first=((1.8,2),(2.8,6))[args.lane]
    data=json.loads((ROOT/'data/balanced400_discovery.json').read_text())
    suffixes,work=matched_suffixes([r['input_tokens'] for r in data],4)
    jobs=[]
    for label,axis,guard,compiler,credit in (
            ('NativeStrong','Native',False,'INSTALLED_DEFAULT_COMPILE',0),
            ('NativeMatched','Native',False,'MATCHED_NO_COMPILE',4),
            ('NativeGuard','Native',True,'MATCHED_NO_COMPILE',4),
            ('Layer4Original','Layer',False,'MATCHED_NO_COMPILE',4),
            ('Layer4Guard','Layer',True,'MATCHED_NO_COMPILE',4),
            ('Token4Original','Token',False,'MATCHED_NO_COMPILE',4),
            ('Token4Guard','Token',True,'MATCHED_NO_COMPILE',4),
            ('Hybrid4Original','Hybrid',False,'MATCHED_NO_COMPILE',4),
            ('Hybrid4Guard','Hybrid',True,'MATCHED_NO_COMPILE',4)):
        extra={}
        if axis!='Native':extra['qualification_gate']='C3_continuous_actions_gate_v1.json'
        if axis=='Token':extra.update(suffix_by_request={'q'+str(i):v for i,v in enumerate(suffixes)},aggregate_work_audit=work)
        if axis=='Hybrid':extra.update(hybrid_suffix_fraction=.125,partition_aligned_chunks=True)
        jobs.append(create(f'GuardAB_Discovery200_q{qps}_{label}_v1',axis,'balanced400_discovery.json',
            cut=4 if axis!='Native' else 0,pair=(first,first+1),qps=qps,seed=2026112291,
            execution_family=compiler,continuous_source_credit_limit=credit,d_quiescent_admission=guard,
            source_capsule=capsule,source_capsule_SHA256=sha,discovery_only=True,
            scope='Same-source,Dataset,arrival,cap and client timing. Only D admission changes within each matched pair. '
                'Guarded D executes one request at a time. P remains continuous. This is an admission mechanism control. '
                'The unrestricted compiled Native remains a separate strong baseline. No paired-start CI follows from one start.',**extra))
    manifest=f'quiescent_admission_AB_lane{args.lane}_v1';plan(manifest,jobs)
    print('WAIT_CORRECTNESS_GATE',manifest,flush=True)
    while not gate():
        assert dispatch_allowed()
        time.sleep(5)
    print('ADMISSION_GATE_PASS',manifest,flush=True)
    for name in jobs:assert execute(name),name
    print('AB_DISCOVERY_COMPLETE',manifest,flush=True)


if __name__=='__main__':main()

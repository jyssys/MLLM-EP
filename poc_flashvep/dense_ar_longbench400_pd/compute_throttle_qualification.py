"""Qualify compute-only deferral without serializing all D admission."""
import argparse
import collections
import json
import time
import torch
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .qualification import audit
from .qualification_gate import compare
from .quiescent_admission_qualification import strict_first
from .quality_gate import compare as quality


def no_mixed(name):
    cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
    rows=json.loads((ROOT/'raw'/name/'actual_queries_decode.json').read_text())['rows']
    groups=collections.defaultdict(list)
    for row in rows:groups[row['batch_id']].append(row)
    bad=[]
    for bid,group in groups.items():
        decode=any(r['start']>=r['original_input_tokens'] for r in group)
        prefill=any(r['start']<r['original_input_tokens'] and r['count']>1 for r in group)
        if decode and prefill:bad.append(bid)
    assert not bad,('Meaningful prefill still mixes with decode',name,bad[:10])
    return dict(meaningful_mixed_batches=0,maximum_D_batch=max(r['requests'] for r in rows),
        batches=len(groups),scope='Native one-query prompt recomputation is not a large suffix prefill. '
            'This coverage test does not assign GPU-exclusive cost or prove decode noninferiority.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('lane',type=int,choices=range(4));args=parser.parse_args()
    torch.set_num_threads(1)
    stock=args.lane==3
    axis=('Layer','Token','Hybrid','Token')[args.lane];first=(0,2,4,6)[args.lane]
    variant='StockToken' if stock else axis
    capsule,sha=freeze('compute_only_throttle_qualification_v1_20261121')
    extra=dict(hybrid_suffix_fraction=.125,partition_aligned_chunks=True) if axis=='Hybrid' else {}
    common=dict(cut=4,pair=(first,first+1),correctness=True,d_compute_throttle=True,
        source_capsule=capsule,source_capsule_SHA256=sha,axis_hash_workers=8,
        wall_timeout=3600,serving_timeout=3300,**extra)
    if stock:common.update(token_stock_model=True,execution_family='INSTALLED_DEFAULT_COMPILE')
    jobs=[]
    single_teacher=None
    if stock:
        single_teacher=create('Q19_Throttle_StockToken4_OriginalTeacher_N1_v1','Native','qualification8.json',
            count=1,pair=(first,first+1),correctness=True,serial=True,output=16,fixed=True,
            reference_single=True,token_teacher_cut=4,state_capture=True,axis_state_capture=False,
            numerical_probe=True,numerical_probe_sync=True,kv_layout='HND',
            execution_family='INSTALLED_DEFAULT_COMPILE',source_capsule=capsule,source_capsule_SHA256=sha)
    for n in (1,8):
        jobs.append(create(f'Q19_Throttle_{variant}4_N{n}_v1',axis,'qualification8.json',
            count=n,controlled=True,serial=True,output=16,fixed=True,**common))
    for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
        jobs.append(create(f'Q19_Throttle_{variant}4_C3_Natural{n}_v1',axis,dataset,
            qps=2.8,seed=2026112201,continuous_source_credit_limit=4,
            axis_capture_hidden_last_only=True,gpu_state_digest=True,axis_import_hash_after_model=True,**common))
    write(ROOT/'configs'/f'compute_throttle_qualification_lane{args.lane}_v1.json',dict(jobs=jobs,
        evidence='FROZEN_CORRECTNESS_DESIGN',
        scope='The installed scheduler already has legal prefill-compute deferral. '
            'Force that path on D while retaining remote-KV loads and continuous decode batching. '
            'This distinguishes compute deferral from the earlier whole-request/device-quiescent admission control. '
            'Require N1,N8,N32 and full400 before clean timing. No allocation counter or assertion is removed.'))
    teacher={'Layer':'Q4_Full36_Teacher_N8','Token':'Q4_TokenTeacher4_N8',
        'Hybrid':'Q4_HybridBoundaryTeacher_f0125_N8'}[axis]
    if stock:
        assert execute(single_teacher),single_teacher
        teacher='Q13_StockFull36_Teacher4_N8_v1'
    results=[]
    for index,name in enumerate(jobs):
        assert execute(name),name
        coverage=audit(name);mix=no_mixed(name)
        assert coverage['passed'] and coverage['decode_full36_observation_complete']
        assert all(v['status']=='PASS' for v in coverage['imported_prefix_KV'])
        if index==0:
            first_proof=strict_first(single_teacher if stock else teacher,name);numerical=dict(passed=all(first_proof.values()),rows=[first_proof])
        elif index==1:numerical=compare(teacher,name)
        else:numerical=quality(name)
        assert numerical['passed'],(name,numerical)
        results.append(dict(name=name,coverage=coverage,compute_deferral=mix,numerical=numerical))
        write(ROOT/'state'/f'D_compute_throttle_{variant}_qualification_v1.json',dict(
            time=time.time(),passed=len(results)==4,full400_passed=len(results)==4,
            axis=axis,stock_token_model=stock,results=results,evidence='CORRECTNESS_ONLY'))
    print('COMPUTE_THROTTLE_QUALIFIED',axis,flush=True)


if __name__=='__main__':main()

"""Test the chunk-residency hypothesis with a common conservative rule."""
import argparse
import json
import time
from .common import ROOT,write,dispatch_allowed
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .geometry import matched_suffixes
from .qualification import audit
from .remote_setup_qualification import native_state
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed
from .metrics import summarize
from .chunk_residency_audit import analyze as chunks


VARIANTS=('Native','Layer','StockToken','Hybrid')


def settings(variant,dataset):
    axis='Token' if variant=='StockToken' else variant
    compiled=variant in ('Native','StockToken')
    value=dict(cut=0 if axis=='Native' else 4 if axis=='Hybrid' else 8,
        execution_family='INSTALLED_DEFAULT_COMPILE' if compiled else 'MATCHED_NO_COMPILE',
        continuous_source_credit_limit=0 if compiled else 4,
        d_compute_throttle=True,defer_D_remote_setup=True)
    if axis!='Native':
        value['qualification_gate']='C3_stock_token_gate_v1.json' if axis=='Token' else 'C3_continuous_actions_gate_v1.json'
    if axis=='Hybrid':
        value.update(hybrid_suffix_fraction=.125,partition_aligned_chunks=True)
    if axis=='Token':
        data=json.loads((ROOT/'data'/dataset).read_text())
        suffix,work=matched_suffixes([r['input_tokens'] for r in data],8)
        value.update(token_stock_model=True,suffix_by_request={'q'+str(i):s for i,s in enumerate(suffix)},
            aggregate_work_audit=work)
    return axis,value


def qualify(lane,capsule,sha):
    variant=VARIANTS[lane]
    first=lane*2
    proofs=[]
    for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
        axis,extra=settings(variant,dataset)
        capture=dict(state_capture=True,axis_state_capture=False) if axis=='Native' else dict(
            axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,axis_hash_workers=8)
        name=create(f'Q38_CompletionGuard_{variant}_Natural{n}_v1',axis,dataset,
            pair=(first,first+1),qps=2.8,seed=2026112201,correctness=True,gpu_state_digest=True,
            D_prefill_completion_guard=True,source_capsule=capsule,source_capsule_SHA256=sha,**extra,**capture)
        assert execute(name),name
        state=native_state(name) if axis=='Native' else audit(name)
        assert state['passed']
        if axis!='Native':
            assert state['decode_full36_observation_complete']
            assert all(v['status']=='PASS' for v in state['imported_prefix_KV'])
        score=quality(name)
        assert score['passed'],(name,score)
        proofs.append(dict(name=name,state=state,quality=score,compute_rule=no_mixed(name)))
        write(ROOT/'state'/f'D_completion_guard_{variant}_gate_v1.json',dict(
            passed=len(proofs)==2,full400_passed=len(proofs)==2,variant=variant,axis=axis,
            cut=extra['cut'],proofs=proofs,evidence='CORRECTNESS_ONLY',
            scope='Inherited independent N1/N8 action-state proofs plus new policy-specific Natural32/400 checks. '
                'The rule changes pre-allocation admission only. Model forward and state-transfer functions are unchanged. '
                'Natural batching can change BI0 outputs. No performance or universal safety claim.'))
    print('COMPLETION_GUARD_QUALIFIED',variant,flush=True)


def discovery(lane,capsule,sha):
    qps,first=((1.8,0),(2.8,4))[lane]
    while not all((ROOT/'state'/f'D_completion_guard_{v}_gate_v1.json').exists() and
        json.loads((ROOT/'state'/f'D_completion_guard_{v}_gate_v1.json').read_text()).get('passed') for v in VARIANTS):
        assert dispatch_allowed()
        time.sleep(5)
    jobs=[]
    for variant in VARIANTS:
        for guard in (False,True):
            axis,extra=settings(variant,'balanced400_discovery.json')
            jobs.append(create(f'CompletionAB_Discovery200_q{qps}_{variant}_Guard{int(guard)}_v1',axis,
                'balanced400_discovery.json',pair=(first,first+1),qps=qps,seed=2026112291,
                D_prefill_completion_guard=guard,discovery_only=True,
                completion_guard_qualification_gate=f'D_completion_guard_{variant}_gate_v1.json' if guard else None,
                source_capsule=capsule,source_capsule_SHA256=sha,
                scope='Within-axis same-source/pair/trace A/B. Change only the partial-prefill admission rule. '
                    'B8192/S1024 and source credit remain unchanged. Keep strong compiled Token8. '
                    'Hybrid4/f1/8 and pure cut8 have nominal matched D rows,with alignment residuals reported. '
                    'Retain regressions. One discovery start is not a serving confidence claim.',**extra))
    write(ROOT/'configs'/f'completion_guard_discovery_lane{lane}_v1.json',dict(jobs=jobs,
        source_SHA256=sha,evidence='FROZEN_CHUNK_ADMISSION_INTERVENTION_DESIGN'))
    for name in jobs:
        assert execute(name),name
        m=summarize(name)
        score=quality(name)
        chunks(name)
        print('COMPLETION_GUARD_DISCOVERY',name,m['TTFT_p95_ms'],m['ITL_p99_ms'],m['TPOT_p95_ms'],
            m['SLO_qualified_output_tokens_per_second'],score['passed'],flush=True)
    write(ROOT/'state'/f'completion_guard_discovery_lane{lane}_complete_v1.json',dict(complete=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('qualify','discovery'))
    parser.add_argument('lane',type=int)
    args=parser.parse_args()
    assert args.lane in (range(4) if args.phase=='qualify' else range(2))
    capsule,sha=freeze('prefill_completion_common_intervention_v1_20261121')
    (qualify if args.phase=='qualify' else discovery)(args.lane,capsule,sha)


if __name__=='__main__':
    main()

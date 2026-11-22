"""Change source credit at a fixed hidden-buffer allocation within each axis."""
import argparse
import json
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .run import execute
from .qualification import audit
from .remote_setup_qualification import native_state
from .compute_throttle_qualification import no_mixed
from .quality_gate import compare as quality
from .metrics import summarize


def extra(axis,dataset):
    result={}
    if axis in ('Layer','Token','Hybrid'):
        result['qualification_gate']='C3_stock_token_gate_v1.json' if axis=='Token' else 'C3_continuous_actions_gate_v1.json'
    if axis=='Token':
        data=json.loads((ROOT/'data'/dataset).read_text())
        suffix,work=matched_suffixes([r['input_tokens'] for r in data],8)
        result.update(token_stock_model=True,suffix_by_request={'q'+str(i):s for i,s in enumerate(suffix)},
            aggregate_work_audit=work)
    if axis=='Hybrid':
        result.update(hybrid_suffix_fraction=.125,partition_aligned_chunks=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('lane',type=int,choices=range(4))
    args=parser.parse_args()
    axis=('Native','Layer','Token','Hybrid')[args.lane]
    first=(0,2,4,6)[args.lane]
    capsule,sha=freeze('source_credit_fixed_pool_sensitivity_hybrid_v1_20261121' if axis=='Hybrid'
        else 'source_credit_fixed_pool_sensitivity_v1_20261121')
    common=dict(cut=0 if axis=='Native' else 4 if axis=='Hybrid' else 8,pair=(first,first+1),
        execution_family='MATCHED_NO_COMPILE' if axis in ('Layer','Hybrid') else 'INSTALLED_DEFAULT_COMPILE',
        d_compute_throttle=True,defer_D_remote_setup=True,
        hidden_source_slots=16,hidden_destination_slots=16,
        hidden_pool_free_reserve_bytes=2*1024**3,
        source_capsule=capsule,source_capsule_SHA256=sha,wall_timeout=1500,serving_timeout=1200)
    write(ROOT/'configs'/f'source_credit_sensitivity_{axis}_v1.json',dict(
        credits=[1,4,8,16],hidden_slots_fixed=16,QPS=2.8,seed=2026112291,
        attempted32='Separate actual-memory preflight for Layer. Do not lower KV42GiB or bypass a failed reserve check.',
        source_SHA256=sha,evidence='FROZEN_CREDIT_CAUSAL_DESIGN',
        scope='Within one axis,only Source Credit changes. Hidden-pool capacity remains16. '
            'Token and Native do not allocate an unused hidden pool. This difference is explicit. '
            'Credit1 is a serialization control,not a default serving candidate. '
            'Credit is acquired at P dispatch and released at the first client output. It is not actual batch size. '
            'Require Natural32/400 lifetime and state checks before timing. Retain failures.'))
    if axis=='Layer':
        settings=dict(common,hidden_source_slots=32,hidden_destination_slots=32,
            continuous_source_credit_limit=32)
        name=create('Credit32_Layer8_ActualMemoryPreflight_N1_v1','Layer','qualification8.json',
            count=1,output=16,fixed=True,correctness=True,axis_capture_hidden_last_only=True,
            gpu_state_digest=True,**settings)
        result=execute(name)
        write(ROOT/'analysis/credit32_Layer8_memory_preflight_v1.json',dict(name=name,complete=result,
            evidence='CORRECTNESS_ONLY' if result else 'FAILED_MEMORY_OR_INITIALIZATION_CHECK',
            scope='Inspect raw hidden_pool_memory receipts before attributing a failure. '
                'No valid latency follows from an initialization or memory-admission failure.'))
    for credit in (1,4,8,16):
        proofs=[]
        gate=ROOT/'state'/f'SourceCredit_{axis}_C{credit}_Pool16_gate_v1.json'
        for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
            capture=dict(state_capture=True,axis_state_capture=False) if axis=='Native' else dict(
                axis_state_capture=True,axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,axis_hash_workers=8)
            name=create(f'Q37_Credit_{axis}_C{credit}_Pool16_Natural{n}_v1',axis,dataset,
                qps=2.8,seed=2026112201,correctness=True,gpu_state_digest=True,
                continuous_source_credit_limit=credit,**common,**capture,**extra(axis,dataset))
            success=execute(name)
            if not success:
                write(gate,dict(passed=False,name=name,evidence='FAILED'))
                break
            state=native_state(name) if axis=='Native' else audit(name)
            assert state['passed']
            if axis!='Native':
                assert state['decode_full36_observation_complete']
                assert all(v['status']=='PASS' for v in state['imported_prefix_KV'])
            score=quality(name)
            assert score['passed'],(name,score)
            proofs.append(dict(name=name,state=state,quality=score,compute_rule=no_mixed(name)))
            write(gate,dict(passed=len(proofs)==2,full400_passed=len(proofs)==2,credit=credit,
                hidden_slots=16,axis=axis,cut=common['cut'],proofs=proofs,evidence='CORRECTNESS_ONLY'))
        if len(proofs)!=2:
            continue
        action_label=axis+('4' if axis=='Hybrid' else '8')
        name=create(f'CreditTune_Discovery200_q2.8_{action_label}_C{credit}_Pool16_v1',axis,'balanced400_discovery.json',
            qps=2.8,seed=2026112291,continuous_source_credit_limit=credit,
            source_credit_qualification_gate=gate.name,discovery_only=True,
            credit_serialization_control=credit==1,**common,**extra(axis,'balanced400_discovery.json'))
        assert execute(name),name
        m=summarize(name)
        print('CREDIT_DISCOVERY',axis,credit,m['TTFT_p95_ms'],m['ITL_p99_ms'],m['TPOT_p95_ms'],
            m['SLO_qualified_output_tokens_per_second'],quality(name)['passed'],flush=True)
    write(ROOT/'state'/f'source_credit_sensitivity_{axis}_complete_v1.json',dict(complete=True))
    print('SOURCE_CREDIT_COMPLETE',axis,flush=True)


if __name__=='__main__':
    main()

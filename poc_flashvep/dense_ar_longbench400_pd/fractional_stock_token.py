"""Test strong pure-token actions at the interior hybrid work levels."""
import argparse
from fractions import Fraction
import json
import time
import torch
from .common import ROOT, write, dispatch_allowed
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .run import execute
from .qualification import audit
from .qualification_gate import compare
from .quiescent_admission_qualification import strict_first
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed
from .remote_setup_qualification import assemble
from .metrics import summarize


def boundary(dataset, effective):
    data=json.loads((ROOT/'data'/dataset).read_text())
    suffix, work=matched_suffixes([r['input_tokens'] for r in data], effective)
    # JSON preserves the rational definition separately. These audit numbers
    # retain their signed lattice residual, rather than asserting exact work.
    work={k: float(v) if isinstance(v,Fraction) else v for k,v in work.items()}
    return dict(suffix_by_request={'q'+str(i):s for i,s in enumerate(suffix)},
        aggregate_work_audit=work, fractional_token_work=[effective.numerator,effective.denominator])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('lane',type=int,choices=(0,1))
    args=parser.parse_args()
    torch.set_num_threads(1)
    first=(2,6)[args.lane]
    capsule,sha=freeze('fractional_stock_token_static_v1_20261121')
    actions=[Fraction(25,4),Fraction(39,4),Fraction(27,2)][args.lane::2]
    write(ROOT/'configs'/f'fractional_stock_token_lane{args.lane}_v1.json',dict(
        effective_cuts=[[a.numerator,a.denominator] for a in actions], source_SHA256=sha,
        evidence='FROZEN_STRONG_BASELINE_DISCOVERY_DESIGN',
        scope='Pure Token effective D rows6.25L,9.75L,13.5L match nominal Hybrid2/f1/8, '
            'Hybrid8/f1/16,Hybrid12/f1/16 before alignment. Token8 already covers Hybrid4/f1/8. '
            'Cut4 is only the adapter template. Per-request suffixes define the actual work. '
            'Keep every integer-lattice residual. Use the unchanged original full36 model and compiler. '
            'N1/N8 strict teachers and Natural32/400 state and quality checks precede timing. '
            'No claim of globally optimal Token tuning.'))
    while not assemble():
        assert dispatch_allowed()
        time.sleep(5)
    for effective in actions:
        label=f'TokenEff{effective.numerator}d{effective.denominator}'
        path=ROOT/'state'/f'C3_fractional_{label}_gate_v1.json'
        common=dict(pair=(first,first+1), execution_family='INSTALLED_DEFAULT_COMPILE',
            source_capsule=capsule,source_capsule_SHA256=sha,wall_timeout=4500,serving_timeout=4000)
        capture=dict(correctness=True,gpu_state_digest=True,
            kernel_activity=True,collector_namespace='kernel_only_collector_v3',
            activity_copies=False,operator_correlation=False,runtime_api_activity=False,
            kernel_activity_seconds=7200)
        teacher=create(f'Q34_OriginalFull36_{label}_N8_v1','Native','qualification8.json',
            reference_single=True,serial=True,output=16,fixed=True,
            token_teacher_cut=4,kv_layout='HND',state_capture=True,axis_state_capture=False,
            numerical_probe=True,numerical_probe_sync=True,kernel_activity_roles=['single'],
            **common,**capture,**boundary('qualification8.json',effective))
        assert execute(teacher),teacher
        numerical=[]
        for n in (1,8):
            actual=create(f'Q34_Stock_{label}_N{n}_v1','Token','qualification8.json',cut=4,count=n,
                controlled=True,serial=True,output=16,fixed=True,token_stock_model=True,
                axis_state_capture=True,axis_capture_hidden_last_only=True,
                axis_import_hash_after_model=True,kernel_activity_roles=['prefill'],
                **common,**capture,**boundary('qualification8.json',effective))
            assert execute(actual),actual
            state=audit(actual)
            strict=compare(teacher,actual) if n==8 else dict(
                passed=all(strict_first(teacher,actual).values()))
            numerical.append(dict(name=actual,state=state,strict=strict))
            write(path,dict(passed=False,full400_passed=False,numerical=numerical,
                effective_cut=[effective.numerator,effective.denominator],evidence='CORRECTNESS_ONLY'))
            if not state['passed'] or not strict['passed']:
                print('FRACTIONAL_TOKEN_STRICT_FAILED',label,n,flush=True)
                break
        if len(numerical)!=2 or not all(r['strict']['passed'] for r in numerical):
            continue
        proofs=[]
        for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
            name=create(f'Q34_RemoteStock_{label}_Natural{n}_v1','Token',dataset,cut=4,
                qps=2.8,seed=2026112201,correctness=True,token_stock_model=True,
                continuous_source_credit_limit=0,d_compute_throttle=True,defer_D_remote_setup=True,
                gpu_state_digest=True,axis_state_capture=True,axis_hash_workers=8,
                axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,
                **common,**boundary(dataset,effective))
            assert execute(name),name
            state=audit(name)
            assert state['passed'] and state['decode_full36_observation_complete']
            assert all(v['status']=='PASS' for v in state['imported_prefix_KV'])
            score=quality(name)
            assert score['passed'],(name,score)
            proofs.append(dict(name=name,state=state,quality=score,compute_deferral=no_mixed(name)))
            write(path,dict(passed=len(proofs)==2,full400_passed=len(proofs)==2,
                effective_cut=[effective.numerator,effective.denominator],numerical=numerical,proofs=proofs,
                qualified_axes=['Token'],qualified_actions=[dict(axis='Token',cut=4,suffix_fraction=None)],
                stock_token_model=True,full400_runtime_stress_passed=len(proofs)==2,
                mixed_rows_passed=True,
                scope='The unmodified original full36 model retains its inherited mixed-row implementation proof. '
                    'This action has separate strict boundary teachers and continuous policy-specific Natural32/400 proofs. '
                    'Compute deferral prevents meaningful suffix/decode coexecution. No cross-start bitwise invariance guarantee.',
                evidence='CORRECTNESS_ONLY'))
        for qps in (1.8,2.8):
            name=create(f'FractionalTune_Discovery200_q{qps}_{label}_v1','Token','balanced400_discovery.json',
                cut=4,qps=qps,seed=2026112291,token_stock_model=True,continuous_source_credit_limit=0,
                d_compute_throttle=True,defer_D_remote_setup=True,
                qualification_gate=path.name,fractional_token_gate=path.name,discovery_only=True,
                scope='Stronger fractional pure-Token static baseline. Read actual suffixes and work rows,not template cut4. '
                    'Single discovery start. Final safety requires independent paired starts.',
                **common,**boundary('balanced400_discovery.json',effective))
            assert execute(name),name
            m=summarize(name)
            print('FRACTIONAL_TOKEN_DISCOVERY',label,qps,m['TTFT_p95_ms'],m['ITL_p99_ms'],
                m['TPOT_p95_ms'],m['SLO_qualified_output_tokens_per_second'],quality(name)['passed'],flush=True)
    print('FRACTIONAL_TOKEN_COMPLETE',args.lane,flush=True)


if __name__=='__main__':
    main()

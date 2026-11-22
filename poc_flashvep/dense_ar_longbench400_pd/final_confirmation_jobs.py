"""Build fresh full-state proofs and paired final continuous-serving starts."""
import hashlib
import json
from functools import cache
from fractions import Fraction
import numpy as np
from .common import ROOT,write,digest
from .experiment import config
from .run import save
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .final_continuous_selection import policy


SIGNATURE_FIELDS=('method','axis_kind','axis_cut','token_stock_model',
    'preserve_compiled_full_model','fuse_short_hybrid','hybrid_suffix_fraction','fractional_token_work',
    'partition_aligned_chunks','whole_prefill_deflection','d_compute_throttle','defer_D_remote_setup',
    'd_quiescent_admission','D_running_admission_cap','execution_family','continuous_source_credit_limit',
    'token_budget','max_num_seqs','KV_block_tokens','hidden_source_slots','hidden_destination_slots',
    'hidden_pool_free_reserve_bytes','hidden_pool_reclaim_unused_cache','D_prefill_completion_guard')


def signature(cfg):
    result={k:cfg.get(k) for k in SIGNATURE_FIELDS}
    for key in ('token_stock_model','preserve_compiled_full_model','fuse_short_hybrid',
            'partition_aligned_chunks','whole_prefill_deflection','d_compute_throttle',
            'defer_D_remote_setup','d_quiescent_admission','hidden_pool_reclaim_unused_cache',
            'D_prefill_completion_guard'):
        result[key]=bool(result[key])
    result['axis_cut']=int(result['axis_cut'] or 0)
    result['continuous_source_credit_limit']=int(result['continuous_source_credit_limit'] or 0)
    # Native and stock Token never allocate these optional transport pools.
    if cfg.get('axis_kind') not in ('Layer','Hybrid'):
        for key in ('hidden_source_slots','hidden_destination_slots','hidden_pool_free_reserve_bytes',
                'hidden_pool_reclaim_unused_cache'):
            result[key]=None
    return result


@cache
def frozen_dataset(dataset):
    path=ROOT/'data'/dataset
    return json.loads(path.read_text()),digest(path)


def clone(base,name,dataset,qps,seed,pair,capsule,sha,correctness=False):
    value=dict(base)
    # These named datasets are frozen. Each actual GPU dispatcher still hashes
    # the file again before execution. Do not parse millions of IDs per clone.
    data,data_sha=frozen_dataset(dataset)
    value.update(name=name,dataset=dataset,dataset_SHA256=data_sha,count=len(data),
        qps=qps,seed=seed,gpus=list(pair),
        arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(1/qps,len(data)-1))].tolist(),
        source_capsule=capsule,source_capsule_SHA256=sha,correctness_only=correctness,
        discovery_only=False,controlled_pipeline=False,ignore_eos=False,output_cap=2048,
        state_capture=correctness and not value.get('axis_kind'),
        axis_state_capture=correctness and bool(value.get('axis_kind')),
        axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,
        gpu_state_digest=correctness,axis_hash_workers=8,
        kernel_activity=False,numerical_probe=False,numerical_probe_sync=False,
        wall_timeout=4500,serving_timeout=4000)
    value.pop('suffix_by_request',None)
    value.pop('aggregate_work_audit',None)
    value.pop('final_composition_gate',None)
    if value.get('axis_kind')=='Token':
        effective=Fraction(*value['fractional_token_work']) if value.get('fractional_token_work') else value['axis_cut']
        suffix,work=matched_suffixes([r['input_tokens'] for r in data],effective)
        work={k:float(v) if isinstance(v,Fraction) else v for k,v in work.items()}
        value.update(suffix_by_request={'q'+str(i):s for i,s in enumerate(suffix)},aggregate_work_audit=work)
    return value


def main():
    if (ROOT/'state/background_final_source_native_control_proofs_v1.json').exists():
        assert (ROOT/'state/final_native_prequalification_complete_v1.json').exists(), 'Drain the owned prequalification before planning duplicate proof keys'
    if (ROOT/'state/background_final_source_candidate_proofs_v1.json').exists():
        assert (ROOT/'state/final_candidate_prequalification_complete_v1.json').exists(), 'Drain the candidate proof cache before final planning'
    if (ROOT/'state/background_compound_credit_cap_tuning_v1.json').exists():
        assert (ROOT/'state/compound_credit_cap_discovery_complete_v1.json').exists(), 'Complete the declared credit/cap interaction before final planning'
    if (ROOT/'state/background_final_source_native_credit_proofs_v1.json').exists():
        assert (ROOT/'state/final_credit_native_prequalification_complete_v1.json').exists(), 'Drain Native credit proof keys before final planning'
    choice=json.loads((ROOT/'configs/final_continuous_static_choices_frozen_v1.json').read_text())
    capsule,sha=freeze('final_continuous_strong_confirmation_v1_20261121')
    prototypes={}
    for qps in (1.8,2.8):
        chosen=choice['selected'][str(qps)]
        entries={'NativeStrong':next(r['NativeStrong'] for r in chosen.values() if r)}
        common={}
        for family,row in chosen.items():
            if row is None:
                continue
            if family=='NativeTuned':
                native_base=json.loads((ROOT/'configs'/(entries['NativeStrong']+'.json')).read_text())
                tuned_base=json.loads((ROOT/'configs'/(row['name']+'.json')).read_text())
                if signature(native_base)==signature(tuned_base):
                    continue
            entries['NativeTuned' if family=='NativeTuned' else 'Best'+family]=row['name']
            key=tuple(row['policy'])
            common[key]=row['NativeCommon']
        extension=ROOT/'configs/final_token_tail_control_extension_v1.json'
        if extension.exists() and qps==2.8:
            extra=json.loads(extension.read_text())
            base=json.loads((ROOT/'configs'/(extra['source_discovery_name']+'.json')).read_text())
            primary=json.loads((ROOT/'configs'/(chosen['Token']['name']+'.json')).read_text())
            for field in ('axis_kind','axis_cut','token_stock_model','execution_family',
                    'continuous_source_credit_limit','d_compute_throttle','defer_D_remote_setup'):
                assert base.get(field)==primary.get(field),('The cap control needs the same primary Token geometry',field)
            assert base['D_running_admission_cap']==2 and primary.get('D_running_admission_cap') is None
            assert not primary.get('D_prefill_completion_guard')
            entries['TokenCap2Control']=extra['source_discovery_name']
            common[policy(base)]=extra['common_native_discovery_name']
        for i,(key,name) in enumerate(sorted(common.items(),key=lambda r:str(r[0]))):
            entries['NativeCommon'+str(i)]=name
        prototypes[str(qps)]={label:json.loads((ROOT/'configs'/(name+'.json')).read_text())
            for label,name in entries.items()}
        prototypes[str(qps)]['Agg']=config('prototype','Agg','longbench_balanced400_v1.json',
            execution_family='INSTALLED_DEFAULT_COMPILE')
    qualification={}
    proof_lanes=[[] for _ in range(4)]
    for cell in prototypes.values():
        for label,base in cell.items():
            if label=='Agg':
                continue
            key=hashlib.sha256(json.dumps(signature(base),sort_keys=True).encode()).hexdigest()[:12]
            if key in qualification:
                continue
            gate='Final_composition_'+key+'_v1.json'
            prior_path=ROOT/'state'/gate
            bridge_path=ROOT/'state'/('C3Bridge_composition_'+key+'_v1.json')
            if not prior_path.exists() and bridge_path.exists():
                bridge=json.loads(bridge_path.read_text())
                if (bridge.get('passed') and bridge.get('full400_passed') and
                        bridge.get('signature')==signature(base) and bridge.get('source_capsule_SHA256')==sha):
                    final_result=bridge['results'][-1]
                    actual=json.loads((ROOT/'raw'/final_result['name']/'configuration.json').read_text())
                    receipt=json.loads((ROOT/'raw'/final_result['name']/'dispatcher_result.json').read_text())
                    assert actual['count']==400 and actual['dataset_SHA256']==digest(ROOT/'data/longbench_balanced400_v1.json')
                    assert actual['correctness_only'] and not actual['ignore_eos'] and signature(actual)==signature(base)
                    assert receipt['complete'] and receipt['source_SHA256']==sha and final_result['quality']['passed']
                    write(prior_path,dict(bridge,name=final_result['name'],
                        adopted_from=bridge_path.name,
                        evidence='NEW_FINAL_SOURCE_FULL400_CONTROL_PROOF_ADOPTED_FROM_BRIDGE',
                        scope='Reuse the exact final-source/signature Natural400 state and quality proof. '
                            'No bridge performance timing selects the main static action. No safety CI is inherited.'))
            if prior_path.exists():
                prior=json.loads(prior_path.read_text())
                if (prior.get('passed') is False and prior.get('source_capsule_SHA256')==sha
                        and prior.get('signature')==signature(base)):
                    qualification[key]=dict(job=prior['name'],gate=gate,signature=signature(base),
                        lane=None,prequalified=True,evidence='FAILED_SOURCE_COMPOSITION_NO_IDENTICAL_RETRY')
                    continue
                if (prior.get('passed') and prior.get('full400_passed') and
                        prior.get('source_capsule_SHA256')==sha and prior.get('signature')==signature(base)):
                    actual=json.loads((ROOT/'raw'/prior['name']/'configuration.json').read_text())
                    receipt=json.loads((ROOT/'raw'/prior['name']/'dispatcher_result.json').read_text())
                    assert actual['count']==400 and actual['dataset_SHA256']==digest(ROOT/'data/longbench_balanced400_v1.json')
                    assert not actual['ignore_eos'] and actual['correctness_only']
                    assert receipt['complete'] and receipt['source_SHA256']==sha and signature(actual)==signature(base)
                    qualification[key]=dict(job=prior['name'],gate=gate,signature=signature(base),
                        lane=None,prequalified=True,evidence='NEW_FINAL_SOURCE_FULL400_CONTROL_PROOF')
                    continue
            lane=len(qualification)%4
            name=f'FinalQ400_{base.get("axis_kind","Native")}_{key}_v1'
            cfg=clone(base,name,'longbench_balanced400_v1.json',2.8,2026112201,
                (2*lane,2*lane+1),capsule,sha,correctness=True)
            save(cfg)
            qualification[key]=dict(job=name,gate=gate,signature=signature(cfg),lane=lane,
                evidence='PENDING_NEW_SOURCE_FULL400_PROOF')
            proof_lanes[lane].append(key)
    performance_lanes=[[] for _ in range(4)]
    cells=[]
    def add(dataset,label,qps,index,seed,first):
        selection_q=1.8 if qps<=1.8 else 2.8
        jobs={}
        for arm,base in prototypes[str(selection_q)].items():
            name=f'FinalC3_{label}_q{qps}_s{index}_{arm}_v1'
            cfg=clone(base,name,dataset,qps,seed,(first,first+1),capsule,sha)
            if arm!='Agg':
                key=hashlib.sha256(json.dumps(signature(base),sort_keys=True).encode()).hexdigest()[:12]
                cfg['final_composition_gate']=qualification[key]['gate']
            cfg.update(final_selection_QPS=selection_q,final_static_choice_SHA256=digest(
                ROOT/'configs/final_continuous_static_choices_frozen_v1.json'),
                scope='Frozen best measured safe-static discovery choices. New held-out or full400 natural performance. '
                    'Original NativeStrong and policy-matched Native controls remain. '
                    'Actual continuous batches and all compile/source-credit/admission differences must be reported. '
                    'No offline oracle,token correction,or time smoothing.')
            save(cfg)
            jobs[arm]=name
        ordered=list(jobs.values())
        np.random.default_rng(seed+int(qps*100)).shuffle(ordered)
        performance_lanes[first//2].extend(ordered)
        cells.append(dict(dataset=dataset,workload=label,QPS=qps,start=index,seed=seed,
            GPU_pair=[first,first+1],jobs=jobs,selection_QPS=selection_q))
    for qps in (1.8,2.8):
        for index,seed in enumerate((2026112411,2026112412,2026112413)):
            add('balanced400_heldout.json','Heldout200',qps,index,seed,0)
    for qps in (1.,1.8,2.8,4.5):
        for index,seed in enumerate((2026112411,2026112412,2026112413) if qps in (1.8,2.8) else (2026112411,)):
            first=(0,2,4)[index] if qps in (1.8,2.8) else 6
            add('longbench_balanced400_v1.json','Main400',qps,index,seed,first)
    path=ROOT/'configs/final_continuous_execution_plan_v1.json'
    assert not path.exists(),'Keep the final execution plan immutable'
    write(path,dict(source_capsule=capsule,source_capsule_SHA256=sha,qualification=qualification,
        qualification_lanes=proof_lanes,performance_lanes=performance_lanes,cells=cells,
        static_choice_SHA256=digest(ROOT/'configs/final_continuous_static_choices_frozen_v1.json'),
        native_control_extension_SHA256=digest(ROOT/'configs/final_native_tuned_control_extension_v1.json'),
        token_tail_control_extension_SHA256=digest(ROOT/'configs/final_token_tail_control_extension_v1.json'),
        evidence='FROZEN_NEW_SOURCE_FINAL_EXECUTION_DESIGN',
        scope='Every selected runtime/action composition receives a new full400 state/lifetime/quality proof. '
            'All paired arms share one new source capsule,exact dataset,arrival trace and GPU pair. '
            'Primary Heldout200 confirmations all use GPUs0/1. '
            'Main400 start0 uses0/1,start1 uses2/3,start2 uses4/5. Endpoint single starts use6/7. '
            'Report hardware/seed blocks and the limited preflight cross-pair scope. '
            'Native policy controls can differ by family. This extends the seven-arm sketch without removing any baseline. '
            'Do not pool requests as independent starts. Do not change choices after held-out results.'))
    print('FROZEN_FINAL_PLAN',len(qualification),'full400 proofs',sum(map(len,performance_lanes)),'serving starts',flush=True)


if __name__=='__main__':
    main()

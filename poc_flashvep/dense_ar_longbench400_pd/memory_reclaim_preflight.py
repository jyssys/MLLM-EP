"""Falsify a raw-free-memory assumption without changing the KV budget."""
import json
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .qualification import audit
from .quiescent_admission_qualification import strict_first
from .quality_gate import compare as quality


def main():
    capsule,sha=freeze('credit32_reclaimable_memory_preflight_v2_20261121')
    write(ROOT/'configs/credit32_reclaimable_memory_preflight_v2.json',dict(
        historical_failure='Credit32_Layer8_ActualMemoryPreflight_N1_v1',
        scope='The v1 check compared raw cuda free memory with pool size plus2GiB. '
            'PyTorch can also hold unused cached allocations. Do not call that check proof of physical infeasibility. '
            'In this independent diagnostic,release unused initialization cache,then apply the same2GiB reserve. '
            'Keep KV42GiB,weights,precision and every live tensor unchanged. '
            'N1 then Natural32 only. No primary performance,full400 qualification or decode-safety claim.',
        evidence='FROZEN_MEMORY_COUNTERFACTUAL_DESIGN'))
    common=dict(cut=8,pair=(2,3),correctness=True,execution_family='MATCHED_NO_COMPILE',
        d_compute_throttle=True,defer_D_remote_setup=True,continuous_source_credit_limit=32,
        hidden_source_slots=32,hidden_destination_slots=32,
        hidden_pool_free_reserve_bytes=2*1024**3,hidden_pool_reclaim_unused_cache=True,
        axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,
        gpu_state_digest=True,axis_hash_workers=8,source_capsule=capsule,source_capsule_SHA256=sha,
        wall_timeout=1500,serving_timeout=1200)
    results=[]
    for n,dataset in ((1,'qualification8.json'),(32,'continuous_qualification32_natural.json')):
        extra=dict(count=1,controlled=True,serial=True,output=16,fixed=True) if n==1 else {}
        settings=dict(common,**extra)
        if n==1:
            settings['continuous_source_credit_limit']=0
        name=create(f'Credit32_Layer8_ReclaimedMemory_N{n}_v2','Layer',dataset,
            qps=2.8,seed=2026112201,**settings)
        success=execute(name)
        result=dict(name=name,completed=success)
        folder=ROOT/'raw'/name
        result['memory_checks']={role:json.loads((folder/f'hidden_pool_memory_{role}.json').read_text())
            for role in ('prefill','decode') if (folder/f'hidden_pool_memory_{role}.json').exists()}
        if success:
            state=audit(name)
            result['coverage']=state
            assert state['passed'] and state['decode_full36_observation_complete']
            assert all(v['status']=='PASS' for v in state['imported_prefix_KV'])
            result['strict_teacher']=strict_first('Q4_Full36_Teacher_N8',name) if n==1 else None
            result['quality']=quality(name) if n==32 else None
        results.append(result)
        write(ROOT/'analysis/credit32_reclaimed_memory_preflight_v2.json',dict(results=results,
            evidence='CORRECTNESS_ONLY' if success else 'FAILED_MEMORY_OR_RUNTIME',
            scope='An initialization-cache counterfactual. Keep the original failed raw-memory check. '
                'A successful N1/N32 does not qualify full400 performance or sustainable memory behavior.'))
        print('CREDIT32_RECLAIM_DIAGNOSTIC',n,success,flush=True)
        if not success:
            break


if __name__=='__main__':
    main()

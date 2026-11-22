"""Prove a legal two-wave D continuation against an independent three-wave teacher."""
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


def main():
    torch.set_num_threads(1)
    pool=json.loads((ROOT/'data/mechanism_input0_discovery.json').read_text())
    assert len(pool)>=32 and all(r['input_tokens']<=8192 for r in pool)
    small=ROOT/'data/fused_short_qualification8.json';mixed=ROOT/'data/fused_short_qualification32.json'
    for path,rows in ((small,pool[:8]),(mixed,pool[:32])):
        if path.exists():assert json.loads(path.read_text())==rows
        else:write(path,rows)
    capsule,sha=freeze('fused_short_hybrid_qualification_v1_20261121')
    common=dict(pair=(2,3),correctness=True,source_capsule=capsule,source_capsule_SHA256=sha,
        wall_timeout=4500,serving_timeout=4000)
    teacher=create('Q23_IndependentThreeWave_Hybrid4_ShortN8_v1','Native',small.name,
        serial=True,output=16,fixed=True,reference_single=True,state_capture=True,
        axis_state_capture=False,numerical_probe=True,numerical_probe_sync=True,kv_layout='HND',
        local_hybrid_teacher_m=4,local_hybrid_teacher_fraction=.125,**common)
    jobs=[teacher]
    for n in (1,8):
        jobs.append(create(f'Q23_FusedShort_Hybrid4_N{n}_v1','Hybrid',small.name,cut=4,count=n,
            controlled=True,serial=True,output=16,fixed=True,
            hybrid_suffix_fraction=.125,partition_aligned_chunks=True,fuse_short_hybrid=True,**common))
    for label,dataset in (('ShortNatural32',mixed.name),('MixedNatural400','longbench_balanced400_v1.json')):
        jobs.append(create(f'Q23_FusedShort_Hybrid4_C3_{label}_v1','Hybrid',dataset,cut=4,
            qps=2.8,seed=2026112201,hybrid_suffix_fraction=.125,
            partition_aligned_chunks=True,fuse_short_hybrid=True,continuous_source_credit_limit=4,
            axis_capture_hidden_last_only=True,gpu_state_digest=True,axis_import_hash_after_model=True,**common))
    write(ROOT/'configs/fused_short_hybrid_qualification_v1.json',dict(jobs=jobs,
        evidence='FROZEN_CORRECTNESS_DESIGN',
        DAG='P prefix lower layers -> D suffix lower layers -> D upper layers for the complete legal query frame.',
        original_DAG='P prefix lower -> D prefix upper -> D suffix full36.',
        scope='The existing row-aware forward already supports the fused topological order. '
            'Do not force a boundary at the prefix/suffix split when total prompt length<=B8192. '
            'Every scheduler iteration still respects B8192. Long prompts retain the original aligned path. '
            'No token-layer operation is omitted or repeated. Numeric execution geometry changes and needs its own teacher.'))
    results=[]
    for i,name in enumerate(jobs):
        assert execute(name),name
        if i==0:continue
        coverage=audit(name)
        if i==1:
            first=strict_first(teacher,name);proof=dict(passed=all(first.values()),rows=[first])
        elif i==2:proof=compare(teacher,name)
        else:proof=quality(name)
        assert coverage['passed'] and coverage['decode_full36_observation_complete'] and proof['passed'],(name,proof)
        if i>2:assert all(v['status']=='PASS' for v in coverage['imported_prefix_KV'])
        results.append(dict(name=name,coverage=coverage,numerical=proof))
        write(ROOT/'state/C3_fused_short_hybrid_gate_v1.json',dict(time=time.time(),passed=len(results)==4,
            full400_passed=len(results)==4,mixed_rows_passed=coverage['mixed_batches']>0,
            fused_short_hybrid=True,qualified_axes=['Hybrid'],
            qualified_actions=[dict(axis='Hybrid',cut=4,suffix_fraction=.125)],
            results=results,evidence='CORRECTNESS_ONLY'))
    print('FUSED_SHORT_HYBRID_FULL_STATE_AND400_PASS',flush=True)


if __name__=='__main__':main()

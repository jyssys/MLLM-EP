"""Require original-model teachers before timing the preserved-compile adapter."""
import argparse
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
    parser=argparse.ArgumentParser();parser.add_argument('lane',type=int,choices=(0,1));args=parser.parse_args()
    torch.set_num_threads(1)
    axis=('Layer','Hybrid')[args.lane];first=(0,4)[args.lane]
    capsule,sha=freeze('preserved_compiled_full_qualification_v1_20261121')
    extra=dict(hybrid_suffix_fraction=.125,partition_aligned_chunks=True) if axis=='Hybrid' else {}
    common=dict(cut=4,pair=(first,first+1),correctness=True,preserve_compiled_full_model=True,
        source_capsule=capsule,source_capsule_SHA256=sha,axis_hash_workers=8,
        wall_timeout=3600,serving_timeout=3300,**extra)
    m0=create(f'Q17_CompiledFull_{axis}4_Mode0_N8_v1',axis,'qualification8.json',
        controlled=True,serial=True,output=16,fixed=True,**common)
    teacher_extra=dict(token_teacher_cut=4,teacher_suffix_fraction=.125) if axis=='Hybrid' else {}
    teacher=create(f'Q17_OriginalCompiledTeacher_{axis}4_N8_v1','Native','qualification8.json',
        pair=(first,first+1),correctness=True,serial=True,output=16,fixed=True,
        reference_single=True,state_capture=True,axis_state_capture=False,
        numerical_probe=True,numerical_probe_sync=True,kv_layout='HND',
        execution_family='INSTALLED_DEFAULT_COMPILE',source_capsule=capsule,source_capsule_SHA256=sha,**teacher_extra)
    jobs=[m0,teacher]
    for n in (1,8):
        jobs.append(create(f'Q17_CompiledFull_{axis}4_Mode3_N{n}_v1',axis,'qualification8.json',
            count=n,controlled=True,serial=True,output=16,fixed=True,
            execution_family='INSTALLED_DEFAULT_COMPILE',**common))
    for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
        jobs.append(create(f'Q17_CompiledFull_{axis}4_C3_Natural{n}_v1',axis,dataset,
            qps=2.8,seed=2026112201,execution_family='INSTALLED_DEFAULT_COMPILE',
            continuous_source_credit_limit=4,axis_capture_hidden_last_only=True,
            gpu_state_digest=True,axis_import_hash_after_model=True,**common))
    write(ROOT/'configs'/f'preserved_compile_qualification_{axis}_v1.json',dict(jobs=jobs,
        evidence='FROZEN_CORRECTNESS_DESIGN',
        scope='Restore the original inner model and decoder-layer functions before engine initialization. '
            'First prove Mode0 against the earlier teacher. Then use a separate default-compiled original-model teacher. '
            'A strict mismatch stops this qualification. Do not replace outputs to obtain a pass.'))
    old_teacher='Q4_Full36_Teacher_N8' if axis=='Layer' else 'Q4_HybridBoundaryTeacher_f0125_N8'
    results=[]
    for index,name in enumerate(jobs):
        assert execute(name),name
        if index==1:continue
        coverage=audit(name)
        assert coverage['passed'] and coverage['decode_full36_observation_complete']
        if index==0:proof=compare(old_teacher,name)
        elif index==2:
            row=strict_first(teacher,name);proof=dict(passed=all(row.values()),rows=[row])
        elif index==3:proof=compare(teacher,name)
        else:
            proof=quality(name)
            assert coverage['mixed_batches']>0
            assert all(v['status']=='PASS' for v in coverage['imported_prefix_KV'])
        write(ROOT/'analysis'/(name+'_compiled_qualification.json'),dict(numerical=proof,coverage=coverage))
        assert proof['passed'],(name,proof)
        results.append(dict(name=name,numerical=proof,coverage=coverage))
        write(ROOT/'state'/f'C3_preserved_compile_{axis}_gate_v1.json',dict(
            time=time.time(),passed=len(results)==5,qualified_axes=[axis],
            qualified_actions=[dict(axis=axis,cut=4,suffix_fraction=.125 if axis=='Hybrid' else None)],
            preserved_compiled_full_model=True,mixed_rows_passed=coverage['mixed_batches']>0,
            full400_passed=coverage['n']==400,results=results,evidence='CORRECTNESS_ONLY'))
    print('PRESERVED_COMPILED_MODEL_QUALIFIED',axis,flush=True)


if __name__=='__main__':main()

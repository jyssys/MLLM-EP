"""Separate a compile-path numerical mismatch from a partition-state error."""
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
    capsule,sha=freeze('preserved_compile_matched_teacher_v2_20261121')
    old=f'Q17_CompiledFull_{axis}4_Mode0_N8_v1'
    assert json.loads((ROOT/'analysis'/(old+'_compiled_qualification.json')).read_text())['numerical']['passed']
    teacher_extra=dict(token_teacher_cut=4,teacher_suffix_fraction=.125) if axis=='Hybrid' else {}
    teacher=create(f'Q18_OriginalStageTeacher_{axis}4_N8_v1','Native','qualification8.json',
        pair=(first,first+1),correctness=True,serial=True,output=16,fixed=True,
        reference_single=True,state_capture=True,axis_state_capture=False,
        numerical_probe=True,numerical_probe_sync=True,kv_layout='HND',
        teacher_eager_prefix_compiled_suffix=True,
        execution_family='INSTALLED_DEFAULT_COMPILE',source_capsule=capsule,source_capsule_SHA256=sha,**teacher_extra)
    extra=dict(hybrid_suffix_fraction=.125,partition_aligned_chunks=True) if axis=='Hybrid' else {}
    common=dict(cut=4,pair=(first,first+1),correctness=True,preserve_compiled_full_model=True,
        execution_family='INSTALLED_DEFAULT_COMPILE',source_capsule=capsule,source_capsule_SHA256=sha,
        axis_hash_workers=8,wall_timeout=3600,serving_timeout=3300,**extra)
    jobs=[teacher,create(f'Q18_CompiledFull_{axis}4_Mode3_N8_v1',axis,'qualification8.json',
        controlled=True,serial=True,output=16,fixed=True,**common)]
    for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
        jobs.append(create(f'Q18_CompiledFull_{axis}4_C3_Natural{n}_v1',axis,dataset,
            qps=2.8,seed=2026112201,continuous_source_credit_limit=4,
            axis_capture_hidden_last_only=True,gpu_state_digest=True,axis_import_hash_after_model=True,**common))
    write(ROOT/'configs'/f'preserved_compile_stage_teacher_{axis}_v2.json',dict(jobs=jobs,
        evidence='FROZEN_CORRECTNESS_DESIGN',old_mismatch=f'Q17_CompiledFull_{axis}4_Mode3_N1_v1',
        scope='The first all-compiled teacher differs at prefix layer0. The actual eager-prefix path must use an independent same-path teacher. '
            'The new teacher keeps all original full36 layers and owns every KV locally. It never imports the partition adapter. '
            'No performance result or output is corrected. Preserve both teacher comparisons.'))
    results=[]
    for index,name in enumerate(jobs):
        assert execute(name),name
        if index==0:
            prior=f'Q17_CompiledFull_{axis}4_Mode3_N1_v1'
            first_proof=strict_first(teacher,prior)
            write(ROOT/'analysis'/(prior+'_matched_stage_teacher_comparison.json'),first_proof)
            assert all(first_proof.values()),first_proof
            continue
        coverage=audit(name)
        assert coverage['passed'] and coverage['decode_full36_observation_complete']
        numerical=compare(teacher,name) if index==1 else quality(name)
        assert numerical['passed'],(name,numerical)
        if index>1:
            assert coverage['mixed_batches']>0
            assert all(v['status']=='PASS' for v in coverage['imported_prefix_KV'])
        results.append(dict(name=name,numerical=numerical,coverage=coverage))
        write(ROOT/'state'/f'C3_preserved_compile_{axis}_stage_gate_v2.json',dict(
            time=time.time(),passed=len(results)==3,qualified_axes=[axis],
            qualified_actions=[dict(axis=axis,cut=4,suffix_fraction=.125 if axis=='Hybrid' else None)],
            preserved_compiled_full_model=True,mixed_rows_passed=coverage['mixed_batches']>0,
            full400_passed=coverage['n']==400,results=results,evidence='CORRECTNESS_ONLY'))
    print('PRESERVED_COMPILE_MATCHED_PATH_PASS',axis,flush=True)


if __name__=='__main__':main()

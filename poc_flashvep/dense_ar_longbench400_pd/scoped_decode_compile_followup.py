"""Qualify eager-prefill CUDA norms with unchanged native compiled decode."""
import argparse
import hashlib
import json
import time
import zipfile
from .common import ROOT,SOURCE,write,digest,preserve
from .final_confirmation_jobs import clone
from .registration_deferral_campaign import registration_signature
from .decode_ir_priority_followup import signature as ir_signature
from .run import save,execute


PLAN='scoped_decode_compile_correctness_frozen_v1.json'


def signature(cfg):
    return dict(ir_signature(cfg),D_partial_IR_norm_priority=bool(cfg.get('D_partial_IR_norm_priority')),
        D_eager_prefill_IR_norm=bool(cfg.get('D_eager_prefill_IR_norm')))


def build():
    preserve();path=ROOT/'configs'/PLAN;assert not path.exists()
    prior=json.loads((ROOT/'configs/decode_IR_priority_correctness_frozen_v1.json').read_text())
    for axis in ('Layer','Hybrid'):
        progress=json.loads((ROOT/'state'/f'decode_IR_lane{0 if axis=="Layer" else 1}_progress_v1.json').read_text())
        assert len(progress['results'])>=2 and all(r['passed'] for r in progress['results'][:2])
    base=ROOT/'references/source_capsules'/prior['source_capsule']
    assert digest(base)==prior['source_capsule_SHA256']
    target=base.with_name('scoped_decode_compile_overlay_v1.zip');assert not target.exists()
    with zipfile.ZipFile(base) as archive:files={n:archive.read(n) for n in archive.namelist()}
    worker='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py'
    teacher_module='poc_flashvep/dense_ar_longbench400_pd/teacher_stage_path.py'
    text=files[worker].decode();before='from .decode_compile_ir_options import options'
    assert text.count(before)==1;text=text.replace(before,'from .scoped_compile_options import options')
    before="        elif cfg.get('preserve_compiled_full_model') or (role=='decode' and cfg.get('preserve_compiled_decode_only')):"
    assert text.count(before)==1
    text=text.replace(before,"        elif role=='decode' and cfg.get('D_partial_IR_norm_priority'):\n"
        "            from .scoped_ir_compiled_adapter import ScopedIRCompiledAdapter as Adapter\n"+before)
    files[worker]=text.encode()
    text=files[teacher_module].decode()
    before='        with override_forward_context(context):return original_outer(module,*a,**kw)'
    assert text.count(before)==1
    text=text.replace(before,'        with override_forward_context(context):\n'
        "            if cfg.get('teacher_mode0_IR_norm_priority'):\n"
        '                from vllm.config.kernel import IrOpPriorityConfig\n'
        "                with IrOpPriorityConfig.with_default(['vllm_c','native']).set_priority():\n"
        '                    return original_outer(module,*a,**kw)\n'
        '            return original_outer(module,*a,**kw)')
    files[teacher_module]=text.encode()
    added=[]
    for name in ('scoped_ir_compiled_adapter.py','scoped_compile_options.py'):
        entry='poc_flashvep/dense_ar_longbench400_pd/'+name
        files[entry]=(SOURCE/name).read_bytes();added.append(entry)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,files[name])
    sha=digest(target)
    write(target.with_suffix('.json'),dict(SHA256=sha,base_capsule=base.name,base_capsule_SHA256=digest(base),
        changed_existing_files=[worker,teacher_module],added_files=added,
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Scope the official CUDA IR norm provider to eager partial-prefill calls only. '
              'Leave full compiled suffix/decode at the default native provider. '
              'An independent original full36 single-GPU teacher uses the same stage-provider contract. '
              'No imported state,model tensor,output or timestamp correction. Disabled families retain original behavior.'))
    qualification={};teachers=[]
    for lane,axis in enumerate(('Layer','Hybrid')):
        old=prior['qualification'][axis]
        teacher_base=json.loads((ROOT/'configs'/('Q18_OriginalStageTeacher_'+axis+'4_N8_v1.json')).read_text())
        teacher_name=f'ScopedCompileTeacher_{axis}_N8_v1'
        tc=clone(teacher_base,teacher_name,'qualification8.json',20.,2026112201,(6,7),target.name,sha,correctness=True)
        tc.update(serial=True,reference_single=True,continuous_source_credit_limit=0,
            output_cap=16,ignore_eos=True,teacher_eager_prefix_compiled_suffix=True,
            teacher_mode0_IR_norm_priority=True,state_capture=True,axis_state_capture=False,
            gpu_state_digest=False,numerical_probe=True,numerical_probe_sync=True)
        save(tc);teachers.append(teacher_name)
        jobs=[]
        for old_name in old['jobs']:
            cfg=json.loads((ROOT/'configs'/(old_name+'.json')).read_text())
            suffix=old_name.split(f'DecodeIRQ_{axis}_',1)[1]
            name=f'ScopedCompileQ_{axis}_'+suffix
            cfg.update(name=name,source_capsule=target.name,source_capsule_SHA256=sha,
                D_mode0_IR_norm_priority=False,D_partial_IR_norm_priority=True)
            save(cfg);jobs.append(name)
        prototype=json.loads((ROOT/'configs'/(jobs[-1]+'.json')).read_text())
        qualification[axis]=dict(lane=lane,jobs=jobs,teacher=teacher_name,
            gate=f'Scoped_compile_{axis}_gate_v1.json',signature=signature(prototype),
            predecessor_gate=old['gate'],expected_global_IR_priority='native')
    write(path,dict(time=time.time(),source_capsule=target.name,source_capsule_SHA256=sha,
        teachers=teachers,qualification=qualification,
        observation='Global CUDA IR priority restores N1/N8 state for both axes,but its N1 worker-interval '
                    'diagnostic stays close to the unchanged Mode0 control. This is not a serving speedup.',
        hypotheses=['H1: CUDA norms are needed only on eager partial-prefill. Scope that provider and retain native compiled full calls.',
                    'H2: Full adapter/host overhead,not the full decode kernel path,dominates the remaining serving tail.'],
        scope='Prospective source and independent original-model teachers. '
              'Wait for the preceding global-IR tests to finish. Do not interrupt the frozen54. '
              'New N1/N8,mixed32,guarded32 and full400 proofs precede any performance. '
              'Do not inherit a safety CI from global IR or Mode0. No new tolerance or SLO.'))
    print('SCOPED_DECODE_COMPILE_PLAN_FROZEN',sha,flush=True)


def teachers():
    from .metrics import summarize
    plan=json.loads((ROOT/'configs'/PLAN).read_text())
    for marker in plan.get('teacher_predecessor_markers',[]):
        while not (ROOT/'state'/marker).exists():time.sleep(20)
    results=[]
    for name in plan['teachers']:
        assert execute(name),name
        summarize(name)
        receipt=json.loads((ROOT/'raw'/name/'dispatcher_result.json').read_text())
        assert receipt['source_SHA256']==plan['source_capsule_SHA256']
        data=json.loads((ROOT/'raw'/name/'prompt_state_single.json').read_text())
        requests=[json.loads(line) for line in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()]
        assert sorted(r['sample_id'] for r in requests)==list(range(8))
        expected={'q'+str(i) for i in range(8)}
        measured={key for key in data['requests'] if key.startswith('q') and key[1:].isdigit()}
        assert measured==expected
        assert all(len(data['requests'][key]['KV'])==36 for key in expected)
        results.append(dict(name=name,passed=True,source_SHA256=receipt['source_SHA256']))
    all_prompt=all(json.loads((ROOT/'configs'/(n+'.json')).read_text()).get('teacher_eager_all_prompt') for n in plan['teachers'])
    write(ROOT/'state'/plan.get('teacher_marker','scoped_compile_teachers_complete_v1.json'),dict(time=time.time(),passed=True,
        source_capsule_SHA256=plan['source_capsule_SHA256'],
        results=results,evidence='CORRECTNESS_ONLY',
        scope='Original full36 one-GPU models. No partition adapter,NIXL,imported KV or corrected tensor. '+
              ('All prompt stages stay eager/CUDA. Decode retains the original native compiled model.' if all_prompt else
               'The reference uses declared eager-prefix CUDA norms and compiled full-suffix/decode native norms.')))


def run(lane):
    import torch
    from .qualification import audit
    from .qualification_gate import compare
    from .quiescent_admission_qualification import strict_first
    from .quality_gate import compare as quality
    from .compute_throttle_qualification import no_mixed
    from .metrics import summarize
    torch.set_num_threads(1)
    plan=json.loads((ROOT/'configs'/PLAN).read_text())
    axis,item=next((a,r) for a,r in plan['qualification'].items() if r['lane']==lane)
    teacher_marker=plan.get('teacher_marker','scoped_compile_teachers_complete_v1.json')
    while not (ROOT/'state'/item['predecessor_gate']).exists() or not (ROOT/'state'/teacher_marker).exists():
        time.sleep(20)
    assert json.loads((ROOT/'state'/teacher_marker).read_text())['passed']
    target=ROOT/'state'/item['gate'];assert not target.exists()
    results=[];failure=None;mixed_pass=False
    for index,name in enumerate(item['jobs']):
        try:
            assert execute(name),name
            state=audit(name)
            assert state['passed'] and state['decode_full36_observation_complete']
            assert all(r['status']=='PASS' for r in state['imported_prefix_KV'])
            if index==0:
                numerical=strict_first(item['teacher'],name);passed=all(numerical.values())
            elif index==1:
                numerical=compare(item['teacher'],name);passed=numerical['passed']
            else:
                numerical=quality(name);passed=numerical['passed']
            if index==2:
                assert state['mixed_batches']>0
                mixed_pass=True
            runtime=json.loads((ROOT/'raw'/name/'actual_runtime_decode.json').read_text())
            assert runtime['compile_mode']=='3' and "rms_norm=['native']" in runtime['kernel_IR_op_priority']
            events=json.loads((ROOT/'raw'/name/'axis_events_decode.json').read_text())['rows']
            assert any(r['event']==plan.get('scope_event','SCOPED_PARTIAL_IR_PRIORITY') for r in events)
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            rule=no_mixed(name) if cfg.get('d_compute_throttle') else None
            results.append(dict(name=name,state=state,numerical=numerical,passed=passed,compute_rule=rule))
            summarize(name);assert passed,(name,numerical)
        except Exception as error:
            failure=dict(name=name,status='FAILED',error=repr(error));break
        write(ROOT/'state'/f'{plan.get("state_prefix","scoped_compile")}_lane{lane}_progress_v1.json',dict(results=results))
        print('SCOPED_COMPILE_QUALIFICATION_PASS',name,flush=True)
    passed=failure is None and len(results)==len(item['jobs'])
    stopped=[dict(name=n,status='NOT_RUN',reason='BLOCKED_CORRECTNESS')
        for n in item['jobs'][len(results):] if failure is None or n!=failure['name']]
    final_cfg=json.loads((ROOT/'configs'/(item['jobs'][-1]+'.json')).read_text())
    write(target,dict(time=time.time(),passed=passed,full400_passed=passed,results=results,
        failure=failure,stopped=stopped,source_capsule_SHA256=plan['source_capsule_SHA256'],
        signature=registration_signature(final_cfg),role_signature=item['signature'],
        qualified_axes=[axis] if passed else [],qualified_actions=[dict(axis=axis,cut=final_cfg['axis_cut'],
            suffix_fraction=final_cfg.get('hybrid_suffix_fraction') if axis=='Hybrid' else None)],
        mixed_rows_passed=mixed_pass,evidence='CORRECTNESS_ONLY' if passed else 'FAILED'))
    print('SCOPED_COMPILE_TERMINAL',axis,passed,failure,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','teachers','run'))
    p.add_argument('--lane',type=int,choices=(0,1),default=0);args=p.parse_args()
    if args.action=='build':build()
    elif args.action=='teachers':teachers()
    else:run(args.lane)

"""Freeze the final whole-prompt-eager/native-decode repair before timings."""
import argparse
import hashlib
import json
import time
import zipfile
from .common import ROOT,SOURCE,write,digest,preserve
from .run import save
from .scoped_decode_compile_followup import signature


QPLAN='eager_prefill_native_decode_correctness_frozen_v2.json'
PPLAN='eager_prefill_native_decode_confirmation_frozen_v2.json'


def build():
    preserve()
    assert not (ROOT/'configs'/QPLAN).exists() and not (ROOT/'configs'/PPLAN).exists()
    oldq=json.loads((ROOT/'configs/scoped_decode_compile_correctness_frozen_v1.json').read_text())
    oldp=json.loads((ROOT/'configs/scoped_compile_confirmation_frozen_v1.json').read_text())
    stop=json.loads((ROOT/'state/scoped_compile_confirmation_complete_v1.json').read_text())
    assert len(stop['results'])==21 and all(r['status']=='NOT_RUN' for r in stop['results'])
    base=ROOT/'references/source_capsules'/oldq['source_capsule'];assert digest(base)==oldq['source_capsule_SHA256']
    target=base.with_name('eager_prefill_native_decode_overlay_v2.zip');assert not target.exists()
    with zipfile.ZipFile(base) as archive:files={n:archive.read(n) for n in archive.namelist()}
    worker='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py'
    text=files[worker].decode();before="        elif role=='decode' and cfg.get('D_partial_IR_norm_priority'):"
    assert text.count(before)==1
    text=text.replace(before,"        elif role=='decode' and cfg.get('D_eager_prefill_IR_norm'):\n"
        "            from .eager_prefill_compiled_adapter import EagerPrefillCompiledAdapter as Adapter\n"+before)
    files[worker]=text.encode()
    teacher_file='poc_flashvep/dense_ar_longbench400_pd/teacher_stage_path.py'
    text=files[teacher_file].decode();before="state['eager']=request.num_computed_tokens<length-suffix"
    assert text.count(before)==1
    text=text.replace(before,"state['eager']=request.num_computed_tokens<(length if cfg.get('teacher_eager_all_prompt') else length-suffix)")
    text=text.replace("execution='ORIGINAL_FULL36_EAGER_PREFIX'",
        "execution='ORIGINAL_FULL36_EAGER_PROMPT' if cfg.get('teacher_eager_all_prompt') else 'ORIGINAL_FULL36_EAGER_PREFIX'")
    files[teacher_file]=text.encode()
    added='poc_flashvep/dense_ar_longbench400_pd/eager_prefill_compiled_adapter.py'
    files[added]=(SOURCE/'eager_prefill_compiled_adapter.py').read_bytes()
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,files[name])
    sha=digest(target)
    write(target.with_suffix('.json'),dict(SHA256=sha,base_capsule=base.name,base_capsule_SHA256=digest(base),
        changed_existing_files=[worker,teacher_file],added_files=[added],
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Every valid prompt query stays eager with CUDA IR normalization,including full36 Hybrid suffix. '
              'Decode-only calls retain the original compiled native model. Prompt completion uses actual forward coverage,not future output. '
              'Independent full36 teachers use the same explicit stage contract. No tensor/output/timestamp correction.'))
    q=dict(oldq,time=time.time(),source_capsule=target.name,source_capsule_SHA256=sha,
        teachers=[],qualification={},teacher_marker='eager_compile_teachers_complete_v2.json',
        state_prefix='eager_compile_v2',scope_event='EAGER_PREFILL_IR_PRIORITY',
        teacher_predecessor_markers=['scoped_compile_control_lane0_complete_v1.json'],
        scope='Final bounded correctness repair. Keep all prompt computation at the original eager/CUDA path, '
              'then use native compiled decode. All older failures remain. No performance before strict/mixed/full400 gates.')
    for axis,item in oldq['qualification'].items():
        teacher_cfg=json.loads((ROOT/'configs'/(item['teacher']+'.json')).read_text())
        teacher_name=f'EagerDecodeTeacher_{axis}_N8_v2'
        teacher_cfg.update(name=teacher_name,gpus=[4,5],source_capsule=target.name,source_capsule_SHA256=sha,
            teacher_eager_all_prompt=True)
        save(teacher_cfg);q['teachers'].append(teacher_name)
        jobs=[]
        for old_name in item['jobs']:
            cfg=json.loads((ROOT/'configs'/(old_name+'.json')).read_text())
            name=old_name.replace('ScopedCompileQ_','EagerDecodeQ_').removesuffix('_v1')+'_v2'
            cfg.update(name=name,source_capsule=target.name,source_capsule_SHA256=sha,D_eager_prefill_IR_norm=True)
            save(cfg);jobs.append(name)
        final=json.loads((ROOT/'configs'/(jobs[-1]+'.json')).read_text())
        q['qualification'][axis]=dict(item,jobs=jobs,teacher=teacher_name,
            gate=f'Eager_decode_{axis}_gate_v2.json',signature=signature(final),
            predecessor_gate=item['gate'])
    write(ROOT/'configs'/QPLAN,q)
    p=dict(oldp,time=time.time(),source_capsule=target.name,source_capsule_SHA256=sha,
        qualification={},proof_lanes=oldp['proof_lanes'],cells=[],performance_jobs=[],
        state_prefix='eager_compile_confirmation_v2',eager_all_prefill=True,
        report_name='dense_ar_longbench400_eager_prefill_native_decode_results.md',
        control_wait_markers=[['eager_compile_teachers_complete_v2.json','scoped_compile_control_lane0_complete_v1.json'],
            ['scoped_compile_control_lane1_complete_v1.json']],
        scope='Versioned replacement before the first old-plan performance start. All21 old planned starts remain NOT_RUN. '
              'All GPU correctness jobs continue. Fresh source controls retain unchanged strong Token and both Native references. '
              'Same original I3/Output16/Cap1,cut,work,seed and0/1 pair. No SLO or output change. '
              'This is the final bounded kernel-path repair,not a new cut/workload search.')
    for arm,item in oldp['qualification'].items():
        if arm.endswith('Scoped'):
            axis=arm.removesuffix('Scoped')
            p['qualification'][arm]=dict(item,gate=q['qualification'][axis]['gate'],
                signature=q['qualification'][axis]['signature'])
        else:
            cfg=json.loads((ROOT/'configs'/(item['job']+'.json')).read_text())
            name=cfg['name'].replace('ScopedSourceQ_','EagerSourceQ_').removesuffix('_v1')+'_v2'
            cfg.update(name=name,source_capsule=target.name,source_capsule_SHA256=sha)
            save(cfg)
            p['qualification'][arm]=dict(item,job=name,gate=f'Eager_source_{arm}_gate_v2.json',signature=signature(cfg))
    for cell in oldp['cells']:
        arms={}
        for arm,old_name in cell['jobs'].items():
            cfg=json.loads((ROOT/'configs'/(old_name+'.json')).read_text())
            name=old_name.replace('ScopedCompile_I3_','EagerDecode_I3_').removesuffix('_v1')+'_v2'
            cfg.update(name=name,source_capsule=target.name,source_capsule_SHA256=sha,
                scope='Same fixed condition and fresh paired0/1 starts. Eager CUDA prefills/native compiled decode. '
                      'All previous failures remain. No cross-source/pair CI,token correction or future-state selection.')
            proof=p['qualification'][arm]
            if cfg.get('defer_D_request_registration'):cfg['registration_deferral_gate']=proof['gate']
            if arm.endswith('Scoped'):
                cfg.update(D_eager_prefill_IR_norm=True,qualification_gate=proof['gate'],D_compile_role_gate=proof['gate'])
            save(cfg);arms[arm]=name
        p['cells'].append(dict(cell,jobs=arms))
    p['performance_jobs']=[n.replace('ScopedCompile_I3_','EagerDecode_I3_').removesuffix('_v1')+'_v2' for n in oldp['performance_jobs']]
    write(ROOT/'configs'/PPLAN,p)
    print('EAGER_PREFILL_NATIVE_DECODE_FROZEN',sha,'21 serving starts',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('build','teachers','qualify','proof','run','analyze'))
    parser.add_argument('--lane',type=int,choices=(0,1),default=0);args=parser.parse_args()
    if args.action=='build':build()
    elif args.action in ('teachers','qualify'):
        from . import scoped_decode_compile_followup as module
        module.PLAN=QPLAN
        module.teachers() if args.action=='teachers' else module.run(args.lane)
    else:
        from . import scoped_compile_confirmation as module
        module.PLAN=PPLAN
        if args.action=='proof':module.proof(args.lane)
        elif args.action=='run':module.run()
        else:module.analyze()

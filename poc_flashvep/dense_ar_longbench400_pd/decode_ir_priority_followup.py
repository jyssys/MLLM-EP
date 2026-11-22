"""Test the observed IR normalization-provider difference without changing P."""
import argparse
import hashlib
import json
import time
import zipfile
from .common import ROOT,SOURCE,write,digest,preserve
from .final_confirmation_jobs import clone
from .registration_deferral_campaign import registration_signature
from .decode_only_compile_followup import role_signature as previous_signature
from .run import save,execute


PLAN='decode_IR_priority_correctness_frozen_v1.json'


def signature(cfg):
    return dict(previous_signature(cfg),D_mode0_IR_norm_priority=bool(cfg.get('D_mode0_IR_norm_priority')))


def build():
    preserve();path=ROOT/'configs'/PLAN;assert not path.exists()
    base=ROOT/'references/source_capsules/decode_only_compile_custom_ops_overlay_v1.zip'
    assert digest(base)=='17021999bdb7436ac256e758f59ecf5e862dfb4c679ea75e15313bb4fd970dd0'
    target=base.with_name('decode_IR_priority_overlay_v1.zip');assert not target.exists()
    with zipfile.ZipFile(base) as archive:files={n:archive.read(n) for n in archive.namelist()}
    worker='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py'
    helper='poc_flashvep/dense_ar_longbench400_pd/decode_compile_ir_options.py'
    before='from .decode_compile_custom_ops import options'
    text=files[worker].decode();assert text.count(before)==1
    text=text.replace(before,'from .decode_compile_ir_options import options')
    before="cudagraph_mode=str(actual.compilation_config.cudagraph_mode),compile_mode=str(actual.compilation_config.mode),"
    assert text.count(before)==1
    text=text.replace(before,before+"\n            kernel_IR_op_priority=str(actual.kernel_config.ir_op_priority),")
    files[worker]=text.encode();files[helper]=(SOURCE/'decode_compile_ir_options.py').read_bytes()
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,files[name])
    sha=digest(target)
    write(target.with_suffix('.json'),dict(SHA256=sha,base_capsule=base.name,base_capsule_SHA256=digest(base),
        changed_existing_files=[worker],added_files=[helper],
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Keep all prior code. Change only the declared D IR normalization priority and record it in ready metadata. '
              'No model tensor,output or timestamp correction. P retains exactly the preceding Mode0 configuration.'))
    prior=json.loads((ROOT/'configs/decode_only_compile_correctness_frozen_v3.json').read_text())
    qualification={}
    for lane,axis in enumerate(('Layer','Hybrid')):
        item=prior['qualification'][axis]
        prototype=json.loads((ROOT/'configs'/(item['jobs'][-1]+'.json')).read_text())
        prototype['D_mode0_IR_norm_priority']=True
        names=[]
        for index,old in enumerate(item['jobs']):
            cfg=json.loads((ROOT/'configs'/(old+'.json')).read_text())
            name=f'DecodeIRQ_{axis}_'+('StrictN1' if index==0 else 'StrictN8' if index==1 else 'Natural32' if index==2 else 'Natural400')+'_v1'
            cfg.update(name=name,source_capsule=target.name,source_capsule_SHA256=sha,D_mode0_IR_norm_priority=True)
            save(cfg);names.append(name)
        mixed=clone(prototype,f'DecodeIRQ_{axis}_MixedNatural32_v1','continuous_qualification32_natural.json',
            20.,2026112201,(lane*2,lane*2+1),target.name,sha,correctness=True)
        mixed.update(d_compute_throttle=False,defer_D_remote_setup=False,defer_D_request_registration=False,
            D_running_admission_cap=None,D_prefill_completion_guard=False)
        save(mixed);names.insert(2,mixed['name'])
        qualification[axis]=dict(lane=lane,jobs=names,teacher=item['teacher'],
            gate=f'Decode_IR_{axis}_gate_v1.json',signature=signature(prototype))
    write(path,dict(time=time.time(),source_capsule=target.name,source_capsule_SHA256=sha,
        qualification=qualification,
        observation='Actual Mode0 logs choose IR rms_norm/fused_add_rms_norm [vllm_c,native]. '
                    'Actual compiled V3 logs choose [native],even with custom_ops all.',
        hypotheses=['H1: IR normalization priority causes the strict D-state difference. '
                    'Explicit Mode0 priority should restore exact prompt KV/hidden.',
                    'H2: Other compiled-model dispatch differences remain. State will still differ after this single intervention.'],
        scope='A separate prospective correctness plan. All V1–V3 failures and the frozen54 remain. '
              'StrictN1/N8,unthrottled Mixed32,then exact guarded Natural32/400 composition. '
              'Require strict state and actual mixed-row coverage before any performance plan. '
              'No new tolerance selected after observation. A strict failure blocks remaining stages.'))
    print('DECODE_IR_PRIORITY_PLAN_FROZEN',sha,flush=True)


def run(lane):
    import torch
    from .qualification import audit
    from .qualification_gate import compare
    from .quiescent_admission_qualification import strict_first
    from .quality_gate import compare as quality
    from .metrics import summarize
    torch.set_num_threads(1)
    plan=json.loads((ROOT/'configs'/PLAN).read_text())
    axis,item=next((a,r) for a,r in plan['qualification'].items() if r['lane']==lane)
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
                assert state['mixed_batches']>0,'The prospective mixed-row proof must actually execute mixed rows'
                mixed_pass=True
            runtime=json.loads((ROOT/'raw'/name/'actual_runtime_decode.json').read_text())
            assert runtime['compile_mode']=='3'
            assert "rms_norm=['vllm_c', 'native']" in runtime['kernel_IR_op_priority']
            results.append(dict(name=name,state=state,numerical=numerical,passed=passed,
                actual_D_IR_priority=runtime['kernel_IR_op_priority']))
            summarize(name)
            assert passed,(name,numerical)
        except Exception as error:
            failure=dict(name=name,status='FAILED',error=repr(error));break
        write(ROOT/'state'/f'decode_IR_lane{lane}_progress_v1.json',dict(results=results))
        print('DECODE_IR_QUALIFICATION_PASS',name,flush=True)
    passed=failure is None and len(results)==len(item['jobs'])
    stopped=[dict(name=n,status='NOT_RUN',reason='BLOCKED_CORRECTNESS')
        for n in item['jobs'][len(results):] if failure is None or n!=failure['name']]
    final_cfg=json.loads((ROOT/'configs'/(item['jobs'][-1]+'.json')).read_text())
    write(target,dict(time=time.time(),passed=passed,full400_passed=passed,results=results,
        failure=failure,stopped=stopped,source_capsule_SHA256=plan['source_capsule_SHA256'],
        signature=registration_signature(final_cfg),role_signature=item['signature'],
        qualified_axes=[axis] if passed else [],qualified_actions=[dict(axis=axis,cut=final_cfg['axis_cut'],
            suffix_fraction=final_cfg.get('hybrid_suffix_fraction') if axis=='Hybrid' else None)],
        mixed_rows_passed=mixed_pass,evidence='CORRECTNESS_ONLY' if passed else 'FAILED',
        scope='Fresh-source role-local compiler and IR-provider proof. No decode-safety or performance claim.'))
    print('DECODE_IR_PRIORITY_TERMINAL',axis,passed,failure,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','run'))
    p.add_argument('--lane',type=int,choices=(0,1),default=0);args=p.parse_args()
    build() if args.action=='build' else run(args.lane)

"""Require a red-capable state gate before timing a new D compiler path."""
import argparse
import fcntl
import hashlib
import json
import time
import zipfile
from .common import ROOT,SOURCE,write,digest,resources,preserve
from .runner import save,execute

def prepare():
    old=ROOT/'references/source_capsules/v2_preemption_observer_fix_v5.zip'
    new=ROOT/'references/source_capsules/v2_compiled_D_partial_v12.zip';assert not new.exists()
    with zipfile.ZipFile(old) as archive:files={n:archive.read(n) for n in archive.namelist()}
    name='poc_flashvep/dense_ar_longbench400_pd/adapter.py';text=files[name].decode()
    anchor="                    local_h,local_r=self.compiled_prefix_segment(module,local_positions,local_h,local_r,begin,end)\n                else:"
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor[:-len('                else:')]+"                elif self.role=='decode' and self.cfg.get('D_compile_partial'):\n"
        "                    from poc_flashvep.dense_ar_output_aware_v2.compiled_d_stage import compiled_D_segment\n"
        "                    local_h,local_r=compiled_D_segment(self,module,local_positions,local_h,local_r,begin,end)\n                else:")
    files[name]=text.encode()
    name='poc_flashvep/dense_ar_output_aware_v2/scoped_prompt_ops.py';text=files[name].decode()
    anchor='is_forward_context_available() and get_forward_context().skip_compiled and'
    assert text.count(anchor)==1
    text=text.replace(anchor,"is_forward_context_available() and (get_forward_context().skip_compiled or getattr(get_forward_context(),'v2_compiled_prompt_cuda_ops',False)) and")
    files[name]=text.encode();files['poc_flashvep/dense_ar_output_aware_v2/compiled_d_stage.py']=(SOURCE/'compiled_d_stage.py').read_bytes()
    with zipfile.ZipFile(new,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            if name.endswith('.py'):compile(data,name,'exec')
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    write(new.with_suffix('.json'),dict(time=time.time(),SHA256=digest(new),parent_SHA256=digest(old),
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Optional compiled D partial intervals. Copy compilation configs and isolate each layer interval/residual signature. '
            'Keep original CUDA prompt primitives under an explicit local context. Keep native compiled decode-only calls unchanged. '
            'No weights,tensors,sampling,attention dependency or SLO are corrected. Fresh strict and mixed-state gates are mandatory.'))
    family=json.loads((ROOT/'configs/c3_family_qualification_v4_schedule_fix1.json').read_text());actions={}
    for axis,label in (('Layer','Layer4'),('Hybrid','Hybrid4_f0125')):
        item=next(r for r in family['items'] if r['label']==label)
        teacher_cfg=json.loads((ROOT/'configs'/(item['teacher']+'.json')).read_text());teacher=f'V2_DCompile_{axis}_OriginalTeacher8_V12'
        teacher_cfg.update(name=teacher,gpus=[2,3],source_capsule=old.name,source_capsule_SHA256=digest(old));save(teacher_cfg)
        original=json.loads((ROOT/'configs'/(item['strict']+'.json')).read_text());jobs=[]
        for compiled,count in ((False,1),(True,1),(True,8)):
            cfg=dict(original);name=f'V2_DCompile_{axis}_{"Compiled" if compiled else "Control"}_N{count}_V12'
            cfg.update(name=name,gpus=[2,3],count=count,source_capsule=new.name,source_capsule_SHA256=digest(new),D_compile_partial=compiled)
            save(cfg);jobs.append(dict(name=name,count=count,compiled=compiled))
        actions[axis]=dict(teacher=teacher,jobs=jobs)
    write(ROOT/'configs/D_partial_compiler_control_v12.json',dict(time=time.time(),source_capsule=new.name,source_capsule_SHA256=digest(new),actions=actions,
        hypotheses=['D partial-prefill launch/dispatch overhead contributes to the observed delay.',
            'Admission and decode residency dominate,so faster D intervals may not improve TTFT.',
            'P service/queue dominates,so a D-only compiler change may leave the serving ranking unchanged.'],
        scope='N1/N8 compiler mechanism gate only. Do not treat these diagnostic host timings as online performance or GPU-exclusive cost.'))
    print('D_PARTIAL_COMPILER_CONTROL_FROZEN',digest(new),flush=True)

def run():
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.qualification import audit
    from ..dense_ar_longbench400_pd.qualification_gate import compare
    from ..dense_ar_longbench400_pd.quiescent_admission_qualification import strict_first
    from .derived_metrics import summarize
    plan=json.loads((ROOT/'configs/D_partial_compiler_control_v12.json').read_text());pair=[2,3];owner=resources();results=[]
    with (ROOT/'state/pair_2_3.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);owner.stop(pair)
        try:
            for axis,item in plan['actions'].items():
                values=[];failure=None
                try:
                    assert execute(item['teacher'])
                    for job in item['jobs']:
                        owner.stop(pair);assert execute(job['name']);summarize(job['name']);coverage=audit(job['name'])
                        strict=strict_first(item['teacher'],job['name']) if job['count']==1 else compare(item['teacher'],job['name'])
                        passed=all(strict.values()) if job['count']==1 else strict['passed']
                        if job['compiled']:
                            events=json.loads((ROOT/'raw'/job['name']/'axis_events_decode.json').read_text())['rows']
                            compiled=[r for r in events if r['event']=='D_COMPILED_PARTIAL_SEGMENT'];assert compiled and all(r['compiled'] for r in compiled)
                        values.append(dict(job,passed=passed,coverage=coverage,numerical=strict));assert passed,strict
                except Exception as error:failure=repr(error)
                result=dict(axis=axis,passed=failure is None,results=values,error=failure,evidence='CORRECTNESS_ONLY' if failure is None else 'FAILED',
                    full400_passed=False,source_capsule_SHA256=plan['source_capsule_SHA256'])
                write(ROOT/'state'/f'D_partial_compiler_{axis}_control_v12.json',result);results.append(result)
                print('D_PARTIAL_COMPILER_GATE',axis,result['passed'],failure,flush=True)
        finally:owner.restore(pair);preserve()
    write(ROOT/'state/D_partial_compiler_controls_complete_v12.json',dict(time=time.time(),results=results,complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));args=p.parse_args();globals()[args.action]()

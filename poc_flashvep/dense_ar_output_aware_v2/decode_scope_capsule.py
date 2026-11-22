"""Freeze native decode dispatch plus exact preemption observations."""
import hashlib
import json
import time
import zipfile
from .common import ROOT,SOURCE,write,digest,preserve
from .runner import save

def main():
    preserve();folder=ROOT/'references/source_capsules';source=folder/'v2_p_compiled_prefix_cuda_ops_v3.zip'
    target=folder/'v2_native_decode_scoped_prompt_ops_v4.zip';assert not target.exists()
    with zipfile.ZipFile(source) as archive:files={n:archive.read(n) for n in archive.namelist()}
    for name in ('prefix_compile_options.py',):files['poc_flashvep/dense_ar_output_aware_v2/'+name]=(SOURCE/name).read_bytes()
    # Use the worker's own rooted common module inside the private capsule.
    text=(SOURCE/'scoped_prompt_ops.py').read_text().replace('from .common import ROOT,write','from poc_flashvep.dense_ar_longbench400_pd.common import ROOT,write')
    files['poc_flashvep/dense_ar_output_aware_v2/scoped_prompt_ops.py']=text.encode()
    name='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py';text=files[name].decode()
    old='    from vllm import LLM,SamplingParams';assert text.count(old)==1
    text=text.replace(old,old+'\n    from poc_flashvep.dense_ar_output_aware_v2.scoped_prompt_ops import install as install_scoped_prompt_ops\n    install_scoped_prompt_ops(cfg,role)')
    old='    original_schedule=Scheduler.schedule';assert text.count(old)==1
    text=text.replace(old,"    preemptions=[];original_preempt=Scheduler._preempt_request\n"
        "    def preempted(self,request,timestamp):\n"
        "        preemptions.append(dict(time=time.monotonic(),request_id=request.request_id,computed_before=request.num_computed_tokens,\n"
        "            prompt_tokens=request.num_prompt_tokens,previous_preemptions=request.num_preemptions))\n"
        "        return original_preempt(self,request,timestamp)\n"
        "    Scheduler._preempt_request=preempted\n"
        "    import atexit\n"
        "    atexit.register(lambda:write(folder/f'scheduler_preemptions_{role}.json',dict(rows=preemptions,\n"
        "        scope='Read-only exact original-scheduler preemption events. Recomputed queries remain in geometry metrics.')))\n"+old)
    files[name]=text.encode()
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            if name.endswith('.py'):compile(data,name,'exec')
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    sha=digest(target);write(target.with_suffix('.json'),dict(SHA256=sha,base_SHA256=digest(source),
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Candidate D decode now keeps the original default compiled CustomOp path. Only eager prompts use original CUDA ops. '
            'Independent original-model teachers use the same stage contract. Add read-only exact scheduler-preemption events to every arm. '
            'No scheduling policy,actual batch,SLO,state tolerance or output correction is changed by this capsule. Fresh gates remain required.'))
    prior=json.loads((ROOT/'configs/compiler_prefix_control_v3.json').read_text());actions={}
    for axis,lane in (('Layer',0),('Hybrid',1)):
        item=prior['actions'][axis]
        teacher_cfg=json.loads((ROOT/'configs'/(item['teacher']+'.json')).read_text())
        teacher=f'V2_DecodeScope_{axis}_OriginalTeacher8'
        teacher_cfg.update(name=teacher,gpus=[2*lane,2*lane+1],source_capsule=target.name,source_capsule_SHA256=sha,
            teacher_scoped_prompt_custom_ops=True)
        save(teacher_cfg);jobs=[]
        for job in item['jobs']:
            if not job['experimental']:continue
            cfg=json.loads((ROOT/'configs'/(job['name']+'.json')).read_text());name=job['name'].replace('_CudaOpsV3','_DecodeScopeV4')
            cfg.update(name=name,gpus=[2*lane,2*lane+1],source_capsule=target.name,source_capsule_SHA256=sha,
                D_compiled_custom_ops_all=False,D_partial_IR_norm_priority=False,D_scoped_prompt_custom_ops=True)
            save(cfg);jobs.append(dict(job,name=name))
        actions[axis]=dict(item,lane=lane,teacher=teacher,jobs=jobs,
            teacher_scope='Fresh independent Full36 eager-CUDA prompt/default-native compiled decode teacher;no partition or imported state.')
    write(ROOT/'configs/compiler_prefix_decode_scope_v4.json',dict(time=time.time(),actions=actions,source_capsule=target.name,
        source_capsule_SHA256=sha,state_prefix='compiler_control_decode_scope_v4',
        hypotheses=['H1: Global CUDA custom ops add decoder cost. Scope them to eager prompts and retain original native decode.',
            'H2: Host hooks,registration or transport dominate tails;kernel policy alone will not remove the residual.',
            'H3: Changed batch and residency mediate the intervention;measure them and do not label host deltas exclusive GPU time.'],
        scope='Prospective correctness-preserving kernel-policy intervention. No active-decode prefill insertion or safety claim. '
            'Retain all preceding failures and both successful P-only strict tests.'))
    print('SCOPED_PROMPT_NATIVE_DECODE_INTERVENTION_FROZEN',sha,flush=True)

if __name__=='__main__':main()

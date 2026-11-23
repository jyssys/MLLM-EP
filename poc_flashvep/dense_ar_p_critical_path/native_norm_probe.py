"""Test scoped P RMSNorm dispatch without changing D,work or accuracy gates."""
import hashlib
import json
import subprocess
import time
import zipfile
from .common import ROOT,REPO,PYTHON,write,digest,bind,resources,preserve
from .capsule import replace_once


def build(parent_name):
    parent=ROOT/'references/source_capsules'/parent_name;target=parent.with_name(parent.stem+'_P_native_IR_norm_r1.zip')
    if target.exists():return target
    with zipfile.ZipFile(parent) as archive:files={name:archive.read(name) for name in archive.namelist()}
    key='poc_flashvep/dense_ar_output_aware_v2/prefix_compiled_adapter.py';value=files[key].decode()
    value=replace_once(value,'            local=isolated_prefix_config(config,end)\n',
        '            local=isolated_prefix_config(config,end)\n'
        '            local.compilation_config.cache_dir += "/norm_native" if self.cfg.get("P_native_IR_norm") else "/norm_cuda"\n')
    value=replace_once(value,"IrOpPriorityConfig.with_default(['vllm_c','native']).set_priority():",
        "IrOpPriorityConfig.with_default(['native'] if self.cfg.get('P_native_IR_norm') else ['vllm_c','native']).set_priority():")
    value=replace_once(value,"            query_rows=int(hidden.shape[0]),compiled=compiled,CUDA_graph=False,\n",
        "            query_rows=int(hidden.shape[0]),compiled=compiled,CUDA_graph=False,\n"
        "            P_native_IR_norm=bool(self.cfg.get('P_native_IR_norm')),\n"
        "            scoped_IR_norm_priority=['native'] if self.cfg.get('P_native_IR_norm') else ['vllm_c','native'],\n")
    files[key]=value.encode()
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    write(target.with_suffix('.json'),dict(time=time.time(),SHA256=digest(target),parent_SHA256=digest(parent),
        changed=[key],all_other_bytes_identical=True,files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
        scope='One P-only IR priority switch. Keep custom_ops=all,P/D work,compiler mode,weights,attention and all D code unchanged. '
            'Separate provider-specific prefix cache directories prevent reuse of a control artifact for the candidate. '
            'The existing compiled Stock path is unchanged and remains the primary baseline. '
            'A failed exact state gate is not waived. No hidden,KV or output correction is allowed.'))
    return target


def prepare():
    runner=bind();items=[]
    for label in ('Layer','Hybrid'):
        template=json.loads((ROOT/'configs'/f'PC_Q_{label}_Original_Strict8_ArrivalFix2.json').read_text())
        parent=json.loads((ROOT/'configs'/f'PC_C3_EqualReplay_s0_Original{label}.json').read_text())
        source=build(parent['source_capsule']);jobs=[]
        for enabled in (False,True):
            cfg=dict(template);name=f'PC_PNorm_{label}_{"NativeIR" if enabled else "CUDAIR"}_Strict8'
            cfg.update(name=name,gpus=[4,5],source_capsule=source.name,source_capsule_SHA256=digest(source),
                P_native_IR_norm=enabled,P_event_timing=False,P_critical_trace=True,source_credit_release_on_retirement=False)
            cfg.pop('qualification_gate_v2',None);runner.save(cfg);jobs.append(name)
        items.append(dict(label=label,jobs=jobs,source_capsule_SHA256=digest(source),pair=[4,5]))
    path=ROOT/'jobs/p_norm_ir_strict_v1.json';assert not path.exists()
    write(path,dict(time=time.time(),items=items,
        hypothesis='P forced CUDA RMSNorm dispatch can retain extra named-kernel work despite prefix compilation.',
        observation='In the first qualified C3 capture,Layer2 P RMSNorm used11.51ms/request. Stock8 used0.079ms/request. '
            'Work differs in that comparison. It does not prove an11.43ms causal saving.',
        test='Change only P scoped IR priority on the same source. Keep custom_ops=all and D execution unchanged. '
            'Require exact8 KV,hidden and greedy output before primary timing. Do not relax the frozen accuracy contract.',
        scope='Two strict pairs,not a compiler parameter sweep. Cache isolation is part of correctness. '
            'The runtime-global priority log does not expose a temporary P scope. Save the actual scoped priority in model-visit records.'))
    log=ROOT/'logs/p_norm_ir_strict_v1_controller.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.native_norm_probe','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/p_norm_ir_strict_controller.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def compare(control,candidate):
    import torch
    from .critical_analysis import records
    from ..dense_ar_longbench400_pd.qualification_gate import KV_equal
    paths=[ROOT/'raw'/name for name in (control,candidate)]
    states=[json.loads((path/'axis_events_decode.json').read_text())['states'] for path in paths]
    outputs=[{r['sample_id']:r for r in records(path/'requests.jsonl')} for path in paths];rows=[]
    assert set(outputs[0])==set(outputs[1])
    for index in sorted(outputs[0]):
        rid='q'+str(index);a,b=[value[rid] for value in states]
        hidden=[torch.load(value['tensor_file'],map_location='cpu',weights_only=False)['hidden'] for value in (a,b)]
        rows.append(dict(request_id=rid,KV_exact=KV_equal(a['KV'],b['KV']),hidden_exact=torch.equal(*hidden),
            greedy_exact=outputs[0][index]['output_ids']==outputs[1][index]['output_ids']))
    return dict(control=control,candidate=candidate,rows=rows,
        passed=all(all(r[k] for k in ('KV_exact','hidden_exact','greedy_exact')) for r in rows),
        scope='Same-source,provider-isolated exact8 differential. Numerical differences are not repaired or hidden. '
            'A failure blocks primary performance. It does not prove that a stock native numerical implementation is mathematically invalid.')


def run():
    import fcntl
    runner=bind();owner=resources();plan=json.loads((ROOT/'jobs/p_norm_ir_strict_v1.json').read_text());results=[]
    from ..dense_ar_longbench400_pd.qualification import audit
    with (ROOT/'state/pair_4_5.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            for item in plan['items']:
                result=dict(label=item['label'],passed=False)
                try:
                    for name in item['jobs']:
                        owner.stop([4,5]);assert runner.execute(name),name
                        assert audit(name)['passed']
                    result.update(compare(*item['jobs']))
                except Exception as error:result['error']=repr(error)
                write(ROOT/'state'/('p_norm_ir_'+item['label'].lower()+'_strict8.json'),dict(time=time.time(),**result,evidence='CORRECTNESS_ONLY' if result['passed'] else 'FAILED'))
                results.append(result);print('P_NATIVE_NORM_STRICT',item['label'],result['passed'],flush=True)
        finally:owner.restore([4,5]);preserve()
    write(ROOT/'state/p_norm_ir_strict_complete.json',dict(time=time.time(),results=results))


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(),flush=True)

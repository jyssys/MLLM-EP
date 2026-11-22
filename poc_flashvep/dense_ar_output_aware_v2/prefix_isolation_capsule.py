"""Apply only compiler ownership isolation to the failed prefix prototype."""
import hashlib
import json
import zipfile
from .common import ROOT,SOURCE,write,digest,preserve
from .runner import save

def main():
    preserve();folder=ROOT/'references/source_capsules'
    source=folder/'v2_p_compiled_prefix_experiment_v1.zip';target=folder/'v2_p_compiled_prefix_isolated_v2.zip'
    assert not target.exists()
    with zipfile.ZipFile(source) as archive:files={n:archive.read(n) for n in archive.namelist()}
    name='poc_flashvep/dense_ar_output_aware_v2/prefix_compiled_adapter.py'
    files[name]=(SOURCE/'prefix_compiled_adapter.py').read_bytes()
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            if name.endswith('.py'):compile(data,name,'exec')
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    sha=digest(target);write(target.with_suffix('.json'),dict(SHA256=sha,base_SHA256=digest(source),
        changed_files=['prefix_compiled_adapter.py'],files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Clone only helper compilation state; isolate mutable backend cache and per-depth AOT identity. '
            'Retain all installed assertions,original layers,CUDA prompt norms,KV metadata and D policy. This is an unqualified intervention.'))
    prior=json.loads((ROOT/'configs/compiler_prefix_control_v1.json').read_text());actions={}
    for axis,lane in (('Layer',2),('Hybrid',1)):
        old=prior['actions'][axis];jobs=[]
        for job in old['jobs']:
            if not job['experimental'] and job['count']==8:continue
            cfg=json.loads((ROOT/'configs'/(job['name']+'.json')).read_text())
            name=cfg['name']+'_IsolatedV2'
            cfg.update(name=name,gpus=[lane*2,lane*2+1],source_capsule=target.name,source_capsule_SHA256=sha)
            save(cfg);jobs.append(dict(job,name=name))
        actions[axis]=dict(old,lane=lane,jobs=jobs,
            teacher_scope='Reuse the immutable independent original-model teacher. Teacher source/math is unchanged by the P-only cache patch.')
    write(ROOT/'configs/compiler_prefix_control_v2.json',dict(time=__import__('time').time(),actions=actions,
        source_capsule=target.name,source_capsule_SHA256=sha,state_prefix='compiler_control_isolated_v2',
        parent_failure='Both actual v1 P helpers failed at the first warm request with Expected tensors only,but got int. '
            'The unchanged controls passed strict N1/N8. No performance was dispatched.'))
    print('ISOLATED_PREFIX_INTERVENTION_FROZEN',sha,flush=True)

if __name__=='__main__':main()

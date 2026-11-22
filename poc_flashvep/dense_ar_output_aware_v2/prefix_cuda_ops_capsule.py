"""Test the original custom-op bundle without relaxing the strict state gate."""
import hashlib
import json
import time
import zipfile
from .common import ROOT,SOURCE,write,digest,preserve
from .runner import save

def main():
    preserve();folder=ROOT/'references/source_capsules'
    source=folder/'v2_p_compiled_prefix_isolated_v2.zip';target=folder/'v2_p_compiled_prefix_cuda_ops_v3.zip';assert not target.exists()
    with zipfile.ZipFile(source) as archive:files={n:archive.read(n) for n in archive.namelist()}
    for name in ('prefix_compiled_adapter.py','prefix_compile_options.py'):
        files['poc_flashvep/dense_ar_output_aware_v2/'+name]=(SOURCE/name).read_bytes()
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            if name.endswith('.py'):compile(data,name,'exec')
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    sha=digest(target);write(target.with_suffix('.json'),dict(SHA256=sha,base_SHA256=digest(source),
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='The isolated helper now uses the original CUDA custom-op bundle. Include complete config factors in its private cache identity. '
            'No model tensor,output,precision,SLO or state tolerance changes. This is a custom-op bundle test,not a SiLU-only attribution.'))
    old=json.loads((ROOT/'configs/compiler_prefix_control_v2.json').read_text());actions={}
    for axis,lane in (('Layer',2),('Hybrid',0)):
        item=old['actions'][axis];jobs=[]
        for job in item['jobs']:
            cfg=json.loads((ROOT/'configs'/(job['name']+'.json')).read_text());name=job['name'].replace('_IsolatedV2','_CudaOpsV3')
            cfg.update(name=name,gpus=[2*lane,2*lane+1],source_capsule=target.name,source_capsule_SHA256=sha)
            if job['experimental']:cfg['P_compiled_custom_ops_all']=True
            save(cfg);jobs.append(dict(job,name=name))
        actions[axis]=dict(item,lane=lane,jobs=jobs)
    write(ROOT/'configs/compiler_prefix_control_v3.json',dict(time=time.time(),actions=actions,
        source_capsule=target.name,source_capsule_SHA256=sha,state_prefix='compiler_control_cuda_ops_v3',
        parent_failure='Cache isolation restored actual runtime execution in both axes. Strict N1 failed in KV and hidden,but greedy16 matched. '
            'Layer0 KV matched and later-layer KV differed. The traced graph used native SiLU. Test the original custom-op bundle; retain the failed source and do not relax the gate.'))
    print('P_CUDA_CUSTOM_OP_INTERVENTION_FROZEN',sha,flush=True)

if __name__=='__main__':main()

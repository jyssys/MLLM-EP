"""Test the runtime's supported CUDA-only profiler as an independent remedy."""
import copy
from .common import *
from .initial_campaign import run
from .cuda_only_profile import analyze


def main():
    result=[]
    for n in (4,24):
        clean=f'recovery_huginn_3500_32_clean{n}'
        cfg=json.loads((ROOT/'configs'/(clean+'.json')).read_text())
        cfg.update(name=f'recovery_huginn_3500_32_native_cuda_only_profile{n}_ownerclose300_v2',
            cuda_only_profile=True,capture_role='prefill',P_only_profiler=False,nsys=True,
            matched_clean=clean,pd_shutdown_timeout=300,profile_finalization_timeout=300,timeout=600)
        cfg['native_transport_limits']=dict(cfg.get('native_transport_limits',{}),shutdown_timeout=300)
        target=ROOT/'configs'/(cfg['name']+'.json')
        if target.exists():assert json.loads(target.read_text())==cfg
        else:write(target,cfg)
        okay=run(cfg['name'])
        record=dict(name=cfg['name'],complete=okay)
        if okay:record['analysis']=analyze(cfg['name'],clean)
        result.append(record)
        write(ROOT/'state/native_profile_results.json',result)
        if not okay:break
    write(ROOT/'state/native_profile_campaign.json',dict(stage='COMPLETE_REVIEW_REQUIRED',results=result,time=time.time()))
    restore_if_idle()


if __name__=='__main__':main()

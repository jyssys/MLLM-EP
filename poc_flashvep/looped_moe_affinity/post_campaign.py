"""Run matched native controls and independent-start checks after the main campaign."""
import json
import time
import traceback
from .common import ROOT,MODELS,write
from .launch import launch

def main():
    while True:
        state=ROOT/'research_state.json'
        if state.exists() and json.loads(state.read_text())['status']=='GPU_CAMPAIGN_COMPLETE_ANALYSIS_REQUIRED':break
        time.sleep(20)
    jobs=[
        dict(id='ifm_ouro_native_full_profile_v2',model='ifm_ouro',gpus=[0],phase='native_profile',timeout_s=14400,
            allocator_conf='expandable_segments:True',reuse_native_from='ifm_ouro_native_full_profile_v1'),
        dict(id='ifm_huginn_native_full_profile_v1',model='ifm_huginn',gpus=[0],phase='native_profile',timeout_s=14400,
            skip_cells=[[16,2048]]),
        dict(id='ifm_huginn_native_full_profile_v2',model='ifm_huginn',gpus=[0],phase='native_profile',timeout_s=14400,
            allocator_conf='expandable_segments:True',reuse_native_from='ifm_huginn_native_full_profile_v1'),
        dict(id='foil1_native_full_profile_v1',model='foil1',gpus=[0],phase='native_profile',timeout_s=14400),
    ]
    for start in [1,2]:
        for ep in [4,8]:
            for model in MODELS:
                jobs.append(dict(id=f'{model}_ep{ep}_quiet_verify_start{start}_v1',model=model,
                    gpus=list(range(4,8)) if ep==4 and start==1 else list(range(ep)),phase='paired_verify',start=start,
                    batches=[1],lengths=[128],warmups=10,repeats=20,generation_cap=32,generation_repeats=3,timeout_s=3600))
    for ep in [4,8]:
        for model in MODELS:
            jobs.append(dict(id=f'{model}_ep{ep}_generic_metadata_controls_v1',model=model,
                gpus=list(range(ep)),phase='generic_controls',timeout_s=43200))
    for job in jobs:
        write(ROOT/'post_state.json',dict(status='RUNNING',current=job,time=time.time()))
        print(launch(job),flush=True)
    write(ROOT/'post_state.json',dict(status='GPU_CONTROLS_COMPLETE',time=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'post_state.json',dict(status='FAILED',time=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise

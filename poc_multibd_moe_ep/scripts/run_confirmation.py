"""Frozen held-out with same-process, counterbalanced observer controls."""
import argparse
import os
import subprocess
import json
import re
import datetime
import importlib.metadata
from poc_multibd_moe_ep.contract import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['ep4','tp4','both'],default='ep4')
    p.add_argument('--session',required=True)
    a=p.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9_-]+',a.session)
    os.environ['POC_MULTIBD_SESSION']=a.session
    cases=[3,2,4,1]
    plan=ROOT/f'artifacts/confirmation_{a.backend}_{a.session}_plan.json'
    assert not plan.exists(),'Preserve prior plans; choose a new session identifier'
    write(plan,dict(
        R_order=cases,requests=24,split='held_out',metadata_selection_unchanged=True,
        gpu_mask=MASK,measurement_session=a.session,
        deterministic_ops=os.environ.get('POC_MULTIBD_DETERMINISTIC')=='1',
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        user_override='2026-10-13: use GPUs 4,5,6,7 instead of 2,3,6,7',
        paired_observers=True,clean_repeats=3,timing_repeats=3,
        trace_repeats=1,warmups_per_request=3,
        reason='Separate-process calibration timing varied substantially; observer ON/OFF must be within the same model session.',
        device_binding_unchanged=True,baseline_wrappers_removed_for_clean=True))
    def capture(cmd):
        return subprocess.check_output(cmd,text=True)
    pins=json.loads((ROOT/'artifacts/pins.json').read_text())
    assert capture(['git','-C',str(UPSTREAM),'rev-parse','HEAD']).strip()==UPSTREAM_SHA
    for name,expected in pins['file_sha256'].items():
        assert sha(UPSTREAM/name)==expected,name
    write(ROOT/f'artifacts/environment_{a.session}.json',dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mask=MASK,
        physical_to_logical={str(gpu):rank for rank,gpu in enumerate(GPUS)},
        uuid_mapping=UUIDS,nvidia_smi=capture(['nvidia-smi']),
        topology=capture(['nvidia-smi','topo','-m']),
        gpu_inventory=capture(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv']),
        git_sha=capture(['git','-C',str(PROJECT),'rev-parse','HEAD']).strip(),
        versions={name:importlib.metadata.version(name) for name in
                  ['torch','triton','deep-ep','transformers','vllm','sglang']},
        upstream_sha=UPSTREAM_SHA,checkpoint=MODEL_ID,revision=MODEL_REVISION,
        spec_sha256=sha(SPEC),manifest_sha256=sha(ROOT/'artifacts/frozen_manifest.json'),
        previous_environment='artifacts/environment.json',
        deterministic_ops=os.environ.get('POC_MULTIBD_DETERMINISTIC')=='1',
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_scratch_poison_fill=False,
        hardware_strata_not_pooled=True,other_GPU_users_untouched=True))
    backends=['ep4','tp4'] if a.backend=='both' else [a.backend]
    for backend,r in [(backend,r) for backend in backends for r in cases]:
        target=ROOT/f'artifacts/R{r}_{backend}_held_out_paired_{a.session}'
        assert not target.exists(),'Preserve completed or failed runs; use a new output directory'
        target.mkdir(parents=True)
        command=[PYTHON,'-m','poc_multibd_moe_ep.scripts.run_sweep','--R',str(r),
            '--output',str(target),'--mode','paired' if backend=='ep4' else 'clean',
            '--backend',backend,'--split','held_out','--limit','24',
            '--repeats','3','--warmup','3','--max-tokens','4096']
        print('START',target.name,flush=True)
        with (target/'console.log').open('w') as out:
            proc=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,env=os.environ)
            try:code=proc.wait(timeout=2700)
            except subprocess.TimeoutExpired:
                proc.terminate();raise
        write(target/'process_status.json',dict(exit_code=code))
        print('FINISH',target.name,code,flush=True)
        if code:raise RuntimeError('Bounded confirmation stopped at failure: '+target.name)


if __name__=='__main__':main()

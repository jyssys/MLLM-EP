"""Register prospective source locks and repeated contention controls."""
import copy
import hashlib
import json
import subprocess
import time
from .common import ROOT,PRIOR,SPEC,FORK,UUIDS,MODEL,write,digest

def main():
    ROOT.mkdir(exist_ok=True,parents=True)
    assert not (ROOT/'registration.json').exists(),'Preserve existing registration'
    prior=json.loads((PRIOR/'p0_contention_plan.json').read_text())
    plan=copy.deepcopy(prior)
    for group in plan['groups']:
        if group.get('barrier'):
            group['barrier']='cb_'+group['barrier']
        for job in group['jobs']:
            job['id']='cb_'+job['id']
            job['blocks']=18432;job['budget']=2048
            if job.get('barrier'):
                job['barrier']='cb_'+job['barrier']
    write(ROOT/'p0_contention_plan.json',plan)
    write(ROOT/'gpu_uuid_map.json',UUIDS)
    write(ROOT/'resource_manifest.json',dict(physical_GPUs=list(range(8)),scope='Eight independent single-GPU workers',
        cpu_affinity='16 fixed disjoint visible CPU IDs per GPU: gpu*16 .. gpu*16+15',
        quiet_controls='All eight GPUs are measurement-reserved during solo/2/4/8 calibration. Burners are restored after the control queue.',
        burn_supervisor_roots=[str(ROOT/'state'),str(PRIOR/'state')]))
    write(ROOT/'model_source_locks.json',dict(model=str(MODEL),revision=MODEL.name,
        config_sha256=digest(MODEL/'config.json'),runtime=str(FORK),
        pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=FORK,text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=FORK)).hexdigest()))
    write(ROOT/'registration.json',dict(wall_s=time.time(),spec_sha256=digest(SPEC),parent_commit='d6ba31c7275019867e2aca473818d42d9b1fcbb0',
        preserved_staged_sha256='7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9',
        drift_review_percent=3,clean_lane_max_workers_until_review=1,
        service_overrun_target=.05,exploratory_risk_allowed=True,incremental_goodput_screen_percent=5,TTFT_p95_regression_limit_percent=5,
        budget_ms=[10,20,40,60,80,120,160,240],static_R=4,model_early_exit=False,
        capacity_minimum_offered=1000,capacity_minimum_independent_starts=3,
        no_future_decode_predictor_required=True,old_D1_forbidden=True,
        hypothesis='Joint must improve complete-R4 prompt or serving outcomes over equally informed same-fairness C-only and Mixed.'))
    write(ROOT/'research_state.json',dict(status='RUNNING',phase=0,completed_experiments=0,
        next='Contention calibration plus CPU implementation and correctness tests. No native-vLLM PP work.',online_joint='NOT_RUN'))
    print('Registered',sum(len(g['jobs']) for g in plan['groups']),'calibration jobs')

if __name__=='__main__':
    main()

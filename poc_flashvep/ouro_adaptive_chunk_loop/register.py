"""Register calibration and sparse factorial work before candidate results."""
import csv
import io
import json
import random
import subprocess
from .common import ROOT, REPO, FORK, SPEC, LEGACY, UUIDS, digest, write

def register():
    def command(args,cwd=None):
        return subprocess.check_output(args,cwd=cwd,text=True)
    ROOT.mkdir(exist_ok=True,parents=True)
    lock=dict(spec_sha256=digest(SPEC),parent=command(['git','rev-parse','HEAD'],REPO).strip(),
        original_staged_sha256='7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9',
        GPUs=UUIDS,topology=command(['nvidia-smi','topo','-m']),cpu=command(['lscpu']),
        gpu_inventory=command(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv']),
        initial_processes=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv']),
        runtime_pin=command(['git','rev-parse','HEAD'],FORK).strip(),
        runtime_diff=command(['git','diff','--binary'],FORK),
        legacy_reports={name:digest(LEGACY/name) for name in ['analysis.md','correctness.md','decision_gates.md','benchmark_summary.md','adaptive_joint_design.md']})
    if not (ROOT/'registration_lock.json').exists():
        write(ROOT/'registration_lock.json',lock)
    groups=[]
    # Every physical GPU receives solo, pair, quartet, and eight-worker conditions.
    # Matched context shapes and repeated cohorts are identical across conditions.
    # Eight solo starts happen first. Later rotations reverse device order.
    for startup in range(2):
        levels=(1,2,4,8) if startup==0 else (8,4,2,1)
        for k in levels:
            devices=list(range(8)) if startup==0 else list(reversed(range(8)))
            for n in range(0,8,k):
                selected=devices[n:n+k]
                name=f'p0_s{startup}_parallel{k}_block{n//k}'
                jobs=[dict(id=f'{name}_gpu{gpu}',kind='calibration',gpu=gpu,
                    barrier=name if k>1 else None,parallel_workers=k,startup=startup,
                    shapes=[[2048,4],[8192,8]],repetitions=3,budget=1024,
                    split='CALIBRATION',timeout_s=1800) for gpu in selected]
                groups.append(dict(barrier=name if k>1 else None,jobs=jobs))
    value=dict(gpus=list(range(8)),groups=groups,scope='CONTENTION_CONTROL_NO_IDLE_BURNERS_DURING_QUIET_WINDOWS')
    if not (ROOT/'p0_contention_plan.json').exists():
        write(ROOT/'p0_contention_plan.json',value)
    rule=dict(frozen_before_candidate=True,seed=20261023,
        primary_SLO_registration='After decode-only reference calibration and before candidate heldout runs',
        absolute_TBT_ms=[75,100,125,150,200,250,500],TTFT_s=[2,5,10],relative_decode_multipliers=[5,10,25],
        prefill_C=[64,128,256,512,768,1024,1536,2048],prefill_G=[1,2,3,4],
        surface_core_shapes=[[512,1],[2033,4],[8177,8]],surface_edge_shapes=[[4096,2],[16384,4],[2033,16]],
        quantum_train_validation_test='Disjoint source families, arrival seeds, and process blocks. Never random row split.',
        predictor_false_feasible_max=.05,predictor_safe_quantile=.99,
        bootstrap_draws=5000,bootstrap_seed=20261023,
        calibration_interference_trigger=.03,quality_hidden_rel_l2_legacy_limit=.03,
        quality_pilot=dict(GSM8K=64,MATH500=50,max_new_tokens=4096,temperature=0),
        qualified_capacity=dict(min_requests_per_arm_rate=1000,min_server_starts=3,attainment=.9),
        no_go_rules='Stop affected scheduling paths for real state/race bugs. Finish defensible Phase0-4 if possible. Do not implement A3 without G4 gate.')
    if not (ROOT/'prospective_rules.json').exists():
        write(ROOT/'prospective_rules.json',rule)
    fields=['id','kind','gpu','parallel_workers','startup','split','status']
    path=ROOT/'experiment_matrix.csv'
    if not path.exists():
        with path.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for group in groups:
                for job in group['jobs']:
                    writer.writerow({**{k:job.get(k) for k in fields},'status':'PENDING'})
    write(ROOT/'research_state.json',dict(current_phase=0,status='RUNNING',completed_phases=[],
        NOT_RUN=[1,2,3,4,5,6],active_plan='p0_contention_plan.json',
        next_actions=['Complete contention controls','Reproduce known batched mismatches','Qualify G3 and staged-state lifecycle','Collect registered cost surface'],
        claims='No new adaptive speedup, capacity, or quality claim.'))
    print('REGISTERED',sum(len(g['jobs']) for g in groups),'calibration jobs')

if __name__=='__main__':
    register()

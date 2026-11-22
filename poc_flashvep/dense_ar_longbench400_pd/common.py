"""Preserve the user workspace and freeze the new campaign contract."""
import json
import os
from pathlib import Path
import subprocess
import time
from ..dense_ar_slo_layer_pd import common as foundation

REPO=foundation.REPO
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/dense_ar_longbench400_2d_20261121'
PREVIOUS=REPO/'poc_flashvep/reports/dense_ar_bi0_axis_rootcause_20261120'
PREVIOUS_SOURCE=REPO/'poc_flashvep/dense_ar_bi0_shortanswer_pd'
BASE='23525199eaff164184ed39e6280cd6896b350e77'
BRANCH='flashvep/dense-ar-longbench400-2d-pd-20261121'
START=1795186946.0
PYTHON,MODEL,MODEL_REVISION=foundation.PYTHON,foundation.MODEL,foundation.MODEL_REVISION
QUIET_LOCK=foundation.QUIET_LOCK
write,digest,preservation=foundation.write,foundation.digest,foundation.preservation
REVISION='5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
REPORTS=('dense_ar_bi0_axis_root_cause_final_summary.md',
    'dense_ar_bi0_input_length_axis_crossover.md','dense_ar_bi0_continuous_serving_bridge.md',
    'dense_ar_bi0_historical_layer_winner_reproduction.md','dense_ar_bi0_ready_to_d_queue_root_cause.md',
    'dense_ar_bi0_attention_gemm_axis_crossover.md','dense_ar_bi0_layer_token_sweetspot_knob_final.md',
    'dense_ar_bi0_axis_selector_oracle_analysis.md','dense_ar_matched_axis_crossover_final_summary.md',
    'dense_ar_axis_service_queue_causal_decomposition.md')


def time_available(field='deadline_epoch',budget=None):
    value=budget if budget is not None else json.loads((ROOT/'state/budget.json').read_text())
    if value.get('time_limit_removed'):return True
    cutoff=value.get(field)
    return cutoff is None or time.time()<cutoff


def dispatch_allowed(budget=None):
    return not (ROOT/'state/STOP_DISPATCH.json').exists() and time_available('last_large_dispatch_epoch',budget)


def preserve():
    expected=json.loads((ROOT/'state/preservation.json').read_text())
    assert preservation()==expected['signatures'],'User checkout or index changed'
    for path,sha in expected['prior_source_hashes'].items():assert digest(REPO/path)==sha,path
    for name,sha in expected['prior_report_hashes'].items():assert digest(REPO/'poc_flashvep/reports'/name)==sha,name


def resources():
    from ..dense_ar_native_reproduction.common import resources as previous_resources
    from ..dense_ar_bi0_axis_rootcause.resource_guard import stop
    owner=previous_resources();owner.ROOT,owner.PRIOR=ROOT,PREVIOUS
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PREVIOUS/'burner.log'))
    owner.stop=lambda gpus:stop(owner,gpus)
    return owner


def initialize():
    for part in ('state','configs','data','raw','logs','analysis','figures','references','tools','jobs'):
        (ROOT/part).mkdir(parents=True,exist_ok=True)
    path=ROOT/'state/preservation.json'
    if path.exists():preserve();return
    assert json.loads((PREVIOUS/'state/post_publication_final_check.json').read_text())['passed']
    branch=subprocess.check_output(['git','branch','--list',BRANCH],cwd=REPO,text=True).strip()
    assert not branch,'Use a separate namespace rather than overwrite a prior campaign'
    old_dirs=(PREVIOUS_SOURCE,REPO/'poc_flashvep/dense_ar_bi0_axis_rootcause')
    old_sources={str(p.relative_to(REPO)):digest(p) for folder in old_dirs for p in folder.glob('*') if p.suffix in ('.py','.cc')}
    write(path,dict(time=time.time(),signatures=preservation(),publication_parent=BASE,
        prior_source_hashes=old_sources,prior_report_hashes={name:digest(REPO/'poc_flashvep/reports'/name) for name in REPORTS}))
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    owner=resources();burners=owner.burners();active=owner.processes()
    assert all(os.getsid(r['pid'])==burners.get(r['gpu']) for r in active),'Unowned GPU work remains untouched'
    write(ROOT/'state/budget.json',dict(start_epoch=START,deadline_epoch=START+30*3600,
        last_large_dispatch_epoch=START+28*3600,maximum_hours=30,
        scope='Reserve the last two hours for validation, reports, archive, publication and resource closeout.'))
    write(ROOT/'state/environment_initial.json',dict(time=time.time(),model_revision=MODEL_REVISION,
        GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        active_GPU_processes=active,approved_burner_sessions=burners,
        approved_script_SHA256=digest(Path('/home/esjung/vllm-ep/run_utilize.sh'))))
    write(ROOT/'configs/campaign_contract_v1.json',dict(time=time.time(),base=BASE,branch=BRANCH,
        dataset_name='LongBench-Balanced400-v1',dataset_revision=REVISION,
        input_buckets=[[3000,6000],[6000,10000],[10000,20000],[20000,36000]],
        input_boundaries='Lower inclusive and upper exclusive, except final upper36000 inclusive.',
        natural_output_buckets=[[1,8],[9,24],[25,49],[50,99]],target_per_cell=25,
        natural_output_cap=2048,natural_generation='Greedy BI0,thinking disabled,EOS allowed. No forced short answers.',
        candidate_tasks_initial=['qasper','multifieldqa_en','2wikimqa','hotpotqa','musique','narrativeqa','qmsum'],
        candidate_expansion_order=['triviaqa','multifieldqa_zh','dureader','samsum','multi_news','gov_report'],
        selection_seed=2026112101,
        selection='Audit every feasible original prompt in the declared tasks. Generate natural lengths before selection. '
            'Select by frozen deterministic ordering within cells,not serving performance. No duplicates,filler or new context truncation. '
            'If a cell is short,allocate remaining requests within its input bucket by adjacent output-bin distance,then globally by input-bin distance. '
            'Keep shortages and pool/selected distribution shifts visible.',
        model_revision=MODEL_REVISION,BI=0,precision='Original BF16',B=8192,S=1024,KV_bytes_per_GPU=42*1024**3,
        controlled_outputs=[8,16,64,128],primary_cuts=[2,4,8,12],
        hybrid=dict(m=[0,2,4,8,12],suffix_fractions=[0,1/16,1/8,1/4,1/3],
            P='Prefix tokens0:L-s through lower36-m layers.',
            D='Prefix upper m layers and suffix all36 layers with exact causal dependencies.',
            no_duplicate_token_layer_work=True),
        SLO=dict(TTFT_ms=5000,every_client_ITL_ms=34,request_TPOT_ms=33),
        native_relative_decode_CI_upper_bound=1.05,observer_interference_limit=.05,
        parallel_pair_interference_limit=.05,
        confirmation='Three independent server starts. Up to five for predeclared uncertain representative cells. '
            'Include Layer candidates,Token winners,boundaries and native/no-offload controls. Never repeat until a preferred sign appears.',
        phase_statuses={key:'NOT_RUN' for key in ('dataset','motivation','axis_map','hybrid','continuous','dynamic','heldout','publication')},
        scope='A new campaign. Previous HOLD and NO-GO results retain their original model/workload/runtime scope. No online dynamic insertion before correctness and deadline gates.'))
    write(ROOT/'state/progress.json',dict(time=time.time(),status='ACTIVE',phase='AUDIT_AND_DATASET_CANDIDATE_CENSUS'))
    preserve()


if __name__=='__main__':initialize();print(ROOT)

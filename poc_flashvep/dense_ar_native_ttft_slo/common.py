"""Own a new namespace and preserve all existing source and user state."""
import json
import os
from pathlib import Path
import subprocess
import time
from ..dense_ar_slo_layer_pd.common import REPO, PYTHON, MODEL, MODEL_REVISION, QUIET_LOCK, write, digest, preservation

SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/dense_ar_native_ttft_joint_slo_20261124'
PRIOR=REPO/'poc_flashvep/reports/dense_ar_p_critical_path_20261123'
V2=REPO/'poc_flashvep/reports/dense_ar_longbench400_v2_20261122'
BASE='0e2e062260103feb53bd0afbb0ac964ebb027cd9'
BRANCH='flashvep/dense-ar-native-ttft-joint-slo-20261124'
START=1795449700
REPORT_NAMES=(
    'dense_ar_native_relative_ttft_root_cause.md',
    'dense_ar_joint_slo_failure_attribution.md',
    'dense_ar_native_token_layer_request_trace.md',
    'dense_ar_pd_critical_path_controlled_interventions.md',
    'dense_ar_ttft_slo_regression_final_summary.md')


def preserve():
    value=json.loads((ROOT/'state/preservation.json').read_text())
    assert preservation()==value['signatures'],'User checkout/index changed. Do not repair it automatically.'
    for relative,sha in value['frozen_files'].items():
        assert digest(REPO/relative)==sha,relative


def resources():
    from ..dense_ar_native_reproduction.common import resources as factory
    from ..dense_ar_bi0_axis_rootcause.resource_guard import stop
    owner=factory();owner.ROOT,owner.PRIOR=ROOT,PRIOR
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PRIOR/'burner.log',V2/'burner.log'))
    owner.stop=lambda gpus:stop(owner,gpus)
    return owner


def initialize():
    if (ROOT/'state/preservation.json').exists():preserve();return
    for name in ('state','configs','data','references','analysis','figures','raw','logs','jobs','archives','publication'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','rev-parse',BASE],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','branch','--list',BRANCH],cwd=REPO,text=True).strip()
    frozen={}
    for package in ('dense_ar_p_critical_path','dense_ar_output_aware_v2','dense_ar_longbench400_pd'):
        for path in (REPO/'poc_flashvep'/package).glob('*.py'):frozen[str(path.relative_to(REPO))]=digest(path)
    for name in ('dense_ar_longbench400_v2_c3_continuous_serving.md',
            'dense_ar_longbench400_v2_decode_safety_and_windows.md',
            'dense_ar_longbench400_v2_output_aware_2d_final_summary.md',
            'dense_ar_2d_p_critical_path_final_summary.md','dense_ar_2d_p_queue_root_cause.md',
            'dense_ar_2d_optimized_c3_confirmation.md'):
        path=ROOT.parent/name;frozen[str(path.relative_to(REPO))]=digest(path)
    for name in ('longbench_balanced400_v2.json','balanced400_v2_heldout.json','balanced400_v2_discovery.json'):
        path=V2/'data'/name;frozen[str(path.relative_to(REPO))]=digest(path)
    write(ROOT/'state/preservation.json',dict(time=time.time(),signatures=preservation(),frozen_files=frozen,base=BASE))
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    owner=resources();approved=owner.burners();actual=owner.processes()
    assert all(Path('/proc',str(row['pid'])).stat().st_uid==os.getuid()
        and os.getsid(row['pid'])==approved.get(row['gpu']) for row in actual),'Unknown GPU process stays untouched'
    write(ROOT/'state/environment_initial.json',dict(time=time.time(),GPU_query=subprocess.check_output([
        'nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),active=actual,approved=approved,
        burner_script=str(owner.SCRIPT),burner_script_SHA256=digest(Path(owner.SCRIPT)),
        bash_wrapper='/home/esjung/vllm-ep/run_utilize.sh',
        bash_wrapper_SHA256=digest(Path('/home/esjung/vllm-ep/run_utilize.sh')),model_revision=MODEL_REVISION))
    write(ROOT/'state/budget.json',dict(start_epoch=START,deadline_epoch=START+12*3600,
        last_large_dispatch_epoch=START+10.5*3600,maximum_hours=12,closeout_reserve_hours=1.5,
        scope='Root-cause-first planning ceiling. Do not consume the budget with duplicate sweeps.'))
    write(ROOT/'configs/research_contract.json',dict(time=time.time(),base=BASE,B=8192,S=1024,
        numerical_mode='BI0',precision='Original BF16',KV_GiB_per_GPU=42,KV_block_tokens=128,
        primary_QPS=2.0,model_revision=MODEL_REVISION,TTFT_SLO_ms=5000,every_client_ITL_SLO_ms=34,
        TPOT_SLO_ms=33,relative_decode_CI_upper=1.05,absolute_decode_attainment=.95,observer_max_fraction=.05,
        primary_workload='Unchanged LongBench400-v2 Main400 Natural EOS',confirmation='Heldout192 equal-output N3',
        scope='Reproduce old metrics before new GPU work. Keep old and new families separate. '
            'Do not weaken Stock compilation,change prompts,SLO or natural EOS. Metric counterfactuals are not runtime speedups.'))
    preserve()


if __name__=='__main__':initialize();print(ROOT)

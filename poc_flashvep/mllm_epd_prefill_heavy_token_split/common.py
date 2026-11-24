"""Own this campaign without changing the user checkout or earlier evidence."""
import json
import os
from pathlib import Path
import subprocess
import time
from ..dense_ar_slo_layer_pd.common import digest,write,append,preservation

REPO=Path('/home/esjung/MLLM-EP-github')
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124'
BASE='762265301f43517db94e4b20796b4676c23ad562'
BRANCH='flashvep/mllm-epd-prefill-heavy-token-split-20261124'
START=1795496939
PYTHON=Path('/home/esjung/.venvs/looped-pd-abc/bin/python')
MODEL_ID='Qwen/Qwen2.5-VL-7B-Instruct'
BURNER_RESTART=Path('/home/esjung/vllm-ep/burner_restart_20261124_ttodrwps')
PRIOR=REPO/'poc_flashvep/reports/dense_ar_native_ttft_joint_slo_20261124'
REPORTS=('mllm_epd_prefill_heavy_poc_spec.md','mllm_epd_related_work_and_novelty_audit.md',
    'mllm_prefill_heavy400_dataset_profile.md','mllm_epd_runtime_correctness_report.md',
    'mllm_epd_duetserve_style_motivation.md','mllm_epd_prefill_bottleneck_and_decode_idle.md',
    'mllm_epd_generic_token_split_correctness.md','mllm_epd_generic_token_split_online_serving.md',
    'mllm_epd_prefill_heavy_token_split_final_summary.md')
PRIOR_REPORTS=('dense_ar_longbench400_v2_c3_continuous_serving.md',
    'dense_ar_2d_p_critical_path_final_summary.md','dense_ar_ttft_slo_regression_final_summary.md',
    'dense_ar_native_relative_ttft_root_cause.md','dense_ar_pd_critical_path_controlled_interventions.md')


def preserve():
    proof=json.loads((ROOT/'state/preservation.json').read_text())
    assert preservation()==proof['signatures'],'User checkout/index changed. Do not repair it automatically.'
    for path,sha in proof['frozen_files'].items():assert digest(REPO/path)==sha,path


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    from ..dense_ar_bi0_axis_rootcause.resource_guard import stop
    owner.ROOT,owner.PRIOR=ROOT,PRIOR
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PRIOR/'burner.log',BURNER_RESTART/'burner.log'))
    owner.stop=lambda gpus:stop(owner,gpus)
    return owner


def experiment_deadline():
    return json.loads((ROOT/'state/budget.json').read_text())['large_experiment_cutoff_epoch']


def initialize():
    marker=ROOT/'state/preservation.json'
    if marker.exists():preserve();return
    for name in ('state','configs','data','media','raw','logs','analysis','figures','references','jobs','archives'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','rev-parse',BASE],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','branch','--list',BRANCH],cwd=REPO,text=True).strip()
    assert not any((ROOT.parent/name).exists() for name in REPORTS),'Do not overwrite an existing campaign report'
    frozen={str((ROOT.parent/name).relative_to(REPO)):digest(ROOT.parent/name) for name in PRIOR_REPORTS}
    for package in ('dense_ar_native_ttft_slo','dense_ar_p_critical_path','dense_ar_output_aware_v2','dense_ar_longbench400_pd'):
        frozen.update({str(p.relative_to(REPO)):digest(p) for p in (REPO/'poc_flashvep'/package).glob('*.py')})
    write(marker,dict(time=time.time(),signatures=preservation(),frozen_files=frozen,base=BASE))
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    owner=resources();owned=owner.burners();actual=owner.processes()
    assert all(Path('/proc',str(r['pid'])).stat().st_uid==os.getuid() and os.getsid(r['pid'])==owned.get(r['gpu']) for r in actual)
    write(ROOT/'state/environment_initial.json',dict(time=time.time(),GPUs=subprocess.check_output([
        'nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),approved_burners=owned,active=actual,
        burner_script='/home/esjung/vllm-ep/run_utilize.sh',burner_wrapper_SHA256=digest('/home/esjung/vllm-ep/run_utilize.sh'),
        burner_implementation_SHA256=digest(owner.SCRIPT)))
    write(ROOT/'state/budget.json',dict(start_epoch=START,maximum_wall_hours=20,deadline_epoch=START+20*3600,
        review_epoch=START+17*3600,large_experiment_cutoff_epoch=START+18*3600,closeout_reserve_hours=2,
        scope='A planning ceiling,not required runtime. Prioritize correctness and actual online baselines over unsupported candidate work.'))
    write(ROOT/'configs/research_contract.json',dict(time=time.time(),base=BASE,model=MODEL_ID,precision='Original BF16',
        numerical_mode='BI0',greedy=True,natural_EOS=True,thinking=False,video=False,main_target_count=400,
        primary_GPU_budget=3,B=8192,S=1024,prefix_cache=False,context_limit_initial=32768,output_cap_initial=2048,
        proposed_source_counts=dict(single_image=50,multi_image=100,multi_page=100,long_document=150),
        no_forced_length_quota=True,no_source_or_encoder_modification=True,
        SLO_initial=dict(TTFT_ms=5000,every_client_ITL_ms=34,TPOT_ms=33),SLO_calibration_status='NOT_RUN',
        SLO_relative_CI_upper=1.05,absolute_decode_attainment_min=.95,observer_interference_max=.05,
        serving_gain_min=.05,TTFT_p50_noninferiority_CI_upper=1.05,quality_task_margin_pp=2,
        scope='Image-only exact EPD. Keep actual online encoding,Natural/replay,C0/oracle/C3 and GPU/host scopes separate. '
            'Source reallocation and context handling must freeze before serving performance. Do not force image removal or output lengths.'))
    preserve()


if __name__=='__main__':initialize();print(ROOT)

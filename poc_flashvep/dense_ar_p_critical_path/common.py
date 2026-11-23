"""Own the new campaign namespace and preserve the user workspace."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from ..dense_ar_slo_layer_pd import common as foundation

REPO=foundation.REPO
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/dense_ar_p_critical_path_20261123'
PRIOR=REPO/'poc_flashvep/reports/dense_ar_longbench400_v2_20261122'
LEGACY=REPO/'poc_flashvep/reports/dense_ar_longbench400_2d_20261121'
BASE='0ae6cd973a4544c6dce13387882479b94858e0d0'
BRANCH='flashvep/dense-ar-p-critical-path-20261123'
START=1795403714
PYTHON,MODEL,MODEL_REVISION=foundation.PYTHON,foundation.MODEL,foundation.MODEL_REVISION
REVISION='5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
QUIET_LOCK=foundation.QUIET_LOCK
write,digest,preservation=foundation.write,foundation.digest,foundation.preservation


def preserve():
    saved=json.loads((ROOT/'state/preservation.json').read_text())
    assert preservation()==saved['signatures'],'User checkout/index changed. Do not repair it automatically.'
    for path,sha in saved['frozen_predecessors'].items():assert digest(REPO/path)==sha,path


def resources():
    from ..dense_ar_native_reproduction.common import resources as original
    from ..dense_ar_bi0_axis_rootcause.resource_guard import stop
    owner=original();owner.ROOT,owner.PRIOR=ROOT,PRIOR
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PRIOR/'burner.log',LEGACY/'burner.log'))
    owner.stop=lambda gpus:stop(owner,gpus)
    return owner


def time_available(field='deadline_epoch',budget=None):
    return time.time()<(budget or json.loads((ROOT/'state/budget.json').read_text()))[field]


def dispatch_allowed(budget=None):
    return not (ROOT/'state/STOP_DISPATCH.json').exists() and time_available('last_large_dispatch_epoch',budget)


def execution_allowed(budget=None):
    return not (ROOT/'state/STOP_EXECUTION.json').exists() and time_available('deadline_epoch',budget)


def copy_data(name):
    source=PRIOR/'data'/name;target=ROOT/'data'/name
    assert source.is_file(),source
    if not target.exists():shutil.copyfile(source,target)
    assert digest(target)==digest(source),name
    return target


def initialize():
    for name in ('state','configs','data','raw','logs','analysis','figures','references','jobs','publication'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    if (ROOT/'state/preservation.json').exists():preserve();return
    assert json.loads((PRIOR/'state/post_publication_final_check_v2.json').read_text())['passed']
    assert subprocess.check_output(['git','rev-parse',BASE],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','branch','--list',BRANCH],cwd=REPO,text=True).strip()
    frozen={}
    for directory in ('dense_ar_longbench400_pd','dense_ar_output_aware_v2'):
        for path in (REPO/'poc_flashvep'/directory).glob('*.py'):frozen[str(path.relative_to(REPO))]=digest(path)
    for path in (REPO/'poc_flashvep/reports').glob('dense_ar_longbench400_v2*.md'):
        frozen[str(path.relative_to(REPO))]=digest(path)
    for filename in ('balanced400_v2_discovery.json','balanced400_v2_heldout.json','longbench_balanced400_v2.json'):
        path=PRIOR/'data'/filename;frozen[str(path.relative_to(REPO))]=digest(path)
    write(ROOT/'state/preservation.json',dict(time=time.time(),signatures=preservation(),frozen_predecessors=frozen,base=BASE))
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    owner=resources();active=owner.processes();approved=owner.burners()
    assert all(os.getsid(r['pid'])==approved.get(r['gpu']) for r in active),'Leave unknown GPU work untouched'
    write(ROOT/'state/environment_initial.json',dict(time=time.time(),GPU_inventory=subprocess.check_output([
        'nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),active=active,approved=approved,
        burner_script_SHA256=digest(Path(owner.SCRIPT)),model_revision=MODEL_REVISION))
    write(ROOT/'state/budget.json',dict(start_epoch=START,deadline_epoch=START+10*3600,
        last_large_dispatch_epoch=START+8.5*3600,maximum_hours=10,closeout_reserve_hours=1.5,
        scope='Initial6–10-hour campaign budget. Reallocation requires a recorded decision,not a changed scientific gate.'))
    for name in ('longbench_balanced400_v2.json','balanced400_v2_heldout.json','balanced400_v2_discovery.json','qualification32.json'):
        copy_data(name)
    shutil.copyfile(PRIOR/'configs/c3_quality_contract_v2.json',ROOT/'configs/c3_quality_contract_v2.json')
    write(ROOT/'configs/research_contract.json',dict(time=time.time(),base=BASE,
        dataset_SHA256=digest(ROOT/'data/longbench_balanced400_v2.json'),model_revision=MODEL_REVISION,
        B=8192,S=1024,KV_GiB_per_GPU=42,KV_block_tokens=128,numerical_mode='BI0',precision='Original BF16',
        TTFT_SLO_ms=5000,every_client_ITL_SLO_ms=34,TPOT_SLO_ms=33,
        relative_decode_gate_CI_upper=1.05,absolute_decode_attainment_minimum=.95,observer_max_fraction=.05,
        primary_QPS=2.0,confirmation_starts=3,target_P_critical_path_reduction_fraction=.05,
        candidate_selection='Discovery only. Freeze before new confirmation seeds. Retain original fixed candidates and strong compiled Token.',
        scope='Do not change prompts,output caps,arrival traces or SLO to manufacture an axis win. '
            'Existing best Layer2 and StockToken8 are not iso-work. Keep strong best-vs-best and matched-work mechanism controls separate. '
            'GPU enclosures and host scopes are not exclusive kernel time. No unverified active-decode insertion.'))
    preserve()


def bind():
    """Rebind audited helpers in this process. Never edit predecessor files."""
    initialize();older=importlib.import_module('poc_flashvep.dense_ar_output_aware_v2.common')
    values={key:globals()[key] for key in ('ROOT','SOURCE','BASE','BRANCH','START','PYTHON','MODEL','MODEL_REVISION',
        'REVISION','QUIET_LOCK','write','digest','preservation','preserve','initialize','resources','time_available','dispatch_allowed','execution_allowed')}
    values['PREVIOUS']=LEGACY
    for key,value in values.items():setattr(older,key,value)
    # Some helpers can be imported before binding in a CPU analysis process.
    for name,module in list(sys.modules.items()):
        if name.startswith('poc_flashvep.dense_ar_output_aware_v2.'):
            for key,value in values.items():
                if hasattr(module,key):setattr(module,key,value)
    importlib.import_module('poc_flashvep.dense_ar_output_aware_v2.bridge').bind()
    # CPU readers can be imported before this binding. Keep their existing
    # functions,but route their file access to this campaign namespace.
    for suffix in ('qualification','qualification_gate','decode_cap_discovery','service_accounting',
        'independent_metric_audit','metrics'):
        module=sys.modules.get('poc_flashvep.dense_ar_longbench400_pd.'+suffix)
        if module is not None:
            for key,value in (('ROOT',ROOT),('write',write),('digest',digest)):
                if hasattr(module,key):setattr(module,key,value)
    from .quality import compare
    importlib.import_module('poc_flashvep.dense_ar_output_aware_v2.quality_gate').compare=compare
    runner=importlib.import_module('poc_flashvep.dense_ar_output_aware_v2.runner')
    runner.VARIABLE_FIELDS.update({'P_critical_trace','P_event_timing','P_trace_ROI_seconds','P_trace_delay_seconds',
        'kernel_activity_seconds','coordinator_trace','cache_priming_reference'})
    if not getattr(runner.execute,'_pc_failure_trace',False):
        import functools
        import traceback
        original_execute=runner.execute
        @functools.wraps(original_execute)
        def observed_execute(name):
            try:return original_execute(name)
            except Exception:
                command=ROOT/'raw'/name/'command.json'
                write(ROOT/'state'/('dispatcher_failure_'+name+'_'+str(time.time_ns())+'.json'),
                    dict(time=time.time(),name=name,traceback=traceback.format_exc(),
                        GPU_command_created=command.exists(),
                        phase='EXECUTION_OR_POSTCHECK' if command.exists() else 'PREFLIGHT_BEFORE_GPU_COMMAND',
                        scope='Observe the exception only. Do not change dispatch,resource checks or the failure verdict.'))
                raise
        observed_execute._pc_failure_trace=True
        runner.execute=observed_execute
    return runner


if __name__=='__main__':initialize();print(ROOT)

"""Keep the BI0 campaign separate from completed Native evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from ..dense_ar_slo_layer_pd import common as foundation

REPO = foundation.REPO
SOURCE = Path(__file__).parent
ROOT = REPO / 'poc_flashvep/reports/dense_ar_bi0_shortanswer200_20261120'
PREVIOUS = REPO / 'poc_flashvep/reports/dense_ar_native_pd_real_workload_reproduction_20261119'
BASE = '3a97151dd4e32b3c4e421de36363931f7ad62621'
BRANCH = 'flashvep/dense-ar-bi0-shortanswer-axis-dynamic-20261120'
PYTHON, MODEL, MODEL_REVISION = foundation.PYTHON, foundation.MODEL, foundation.MODEL_REVISION
QUIET_LOCK = foundation.QUIET_LOCK
write, digest, git, preservation = foundation.write, foundation.digest, foundation.git, foundation.preservation


def preserve():
    expected = json.loads((ROOT / 'state/preservation.json').read_text())['signatures']
    assert preservation() == expected


def resources():
    from ..dense_ar_native_reproduction.common import resources as prior_resources
    owner = prior_resources()
    owner.ROOT, owner.PRIOR = ROOT, PREVIOUS
    owner.APPROVED_LOGS.add(ROOT / 'burner.log')
    return owner


def initialize():
    for name in ('state', 'configs', 'data', 'raw', 'logs', 'analysis', 'figures', 'references', 'tools'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    path = ROOT / 'state/preservation.json'
    if path.exists():
        preserve()
        return
    assert json.loads((PREVIOUS / 'state/post_publication_final_check.json').read_text())['passed']
    assert not git('branch', '--list', BRANCH)
    write(path, dict(time=time.time(), signatures=preservation(), publication_parent=BASE))
    subprocess.run(['git', 'branch', BRANCH, BASE], cwd=REPO, check=True)
    owner = resources()
    owned, active = owner.burners(), owner.processes()
    assert all(os.getsid(r['pid']) == owned.get(r['gpu']) for r in active)
    start = time.time()
    write(ROOT / 'state/budget.json', dict(start_epoch=start, deadline_epoch=start+10*3600,
        last_large_dispatch_epoch=start+9*3600, hours=10,
        scope='Final hour is reserved for confirmation, reports, push and resource closeout.'))
    write(ROOT / 'state/environment.json', dict(time=start, model_revision=MODEL_REVISION,
        GPU_inventory=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version', '--format=csv'], text=True),
        approved_burners=owned, active_processes=active))
    reports = ('dense_ar_longbench200_native_pd_aggdp2_final_summary.md',
        'dense_ar_longbench200_dataset_contract.md', 'dense_ar_native_decode_latency_root_cause.md',
        'dense_ar_longbench200_prefill_heavy_feasibility.md',
        'dense_ar_axis_sweetspot_knob_discovery_final_summary.md',
        'dense_ar_axis_service_queue_causal_decomposition.md')
    write(ROOT / 'state/prior_report_hashes.json', {n: digest(REPO/'poc_flashvep/reports'/n) for n in reports})
    source = PREVIOUS / 'references/CUPTI_collector'
    target = ROOT / 'references/CUPTI_collector'
    target.mkdir(exist_ok=True)
    for p in source.iterdir():
        if p.is_file():
            shutil.copyfile(p, target/p.name)
            assert digest(p) == digest(target/p.name)
    write(ROOT / 'state/progress.json', dict(time=start, phase='DATASET_AND_BI0_REFERENCE',
        status='ACTIVE', scope='BI0 only. Online offloading requires independent correctness and safe-window gates.'))
    preserve()


def save(cfg):
    initialize()
    assert cfg['batch_invariant'] is False, 'Never mix BI1 into the BI0 Primary'
    path = ROOT / 'configs' / (cfg['name']+'.json')
    if path.exists():
        assert json.loads(path.read_text()) == cfg
    else:
        write(path, cfg)
    return cfg['name']


if __name__ == '__main__':
    initialize()
    print(ROOT)

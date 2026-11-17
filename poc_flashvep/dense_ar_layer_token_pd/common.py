"""Preserve the checkout, index, installed runtime, and prior evidence."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..duetserve_agg_decode import common as agg
from ..dense_ar_slo_layer_pd import common as prior

REPO = prior.REPO
ROOT = REPO / 'poc_flashvep/reports/dense_ar_layer_token_dynamic_headroom_20261117'
SOURCE = Path(__file__).parent
BASE = 'f0e80395aef488a7b3c6148d4155784f87291df2'
BRANCH = 'flashvep/dense-ar-layer-token-dynamic-headroom-20261117'
PREVIOUS = agg.ROOT
SLO_PRIOR = prior.ROOT
PYTHON = prior.PYTHON
MODEL = prior.MODEL
MODEL_REVISION = prior.MODEL_REVISION
QUIET_LOCK = prior.QUIET_LOCK
write, digest, git, preservation = prior.write, prior.digest, prior.git, prior.preservation


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    owner.ROOT, owner.PRIOR = ROOT, ROOT / 'no_legacy_manifest_scan'
    owner.APPROVED_LOGS.update((ROOT / 'burner.log', PREVIOUS / 'burner.log',
        SLO_PRIOR / 'burner.log', prior.PREVIOUS / 'burner.log', prior.STATIC / 'burner.log'))
    # Older state directories also contain burner_* audit summaries. The
    # inherited guard expects only supervisor receipts. Import valid receipts
    # without changing, deleting, or renaming any prior record.
    for path in (PREVIOUS / 'state').glob('burner_*.json'):
        record = json.loads(path.read_text())
        if isinstance(record, dict) and 'GPUs' in record and 'supervisor' in record:
            target = ROOT / 'state' / ('burner_inherited_' + str(record['supervisor']) + '.json')
            if not target.exists():
                write(target, record)
    return owner


def preserve():
    assert preservation() == json.loads((ROOT / 'state/preservation.json').read_text())['signatures']


def initialize():
    for name in ('state', 'configs', 'raw', 'logs', 'data', 'analysis', 'figures', 'references'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    receipt = ROOT / 'state/preservation.json'
    if receipt.exists():
        preserve()
        return
    assert git('rev-parse', BASE) == BASE
    write(receipt, dict(time=time.time(), signatures=preservation(), publication_parent=BASE))
    subprocess.run(['git', 'branch', BRANCH, BASE], cwd=REPO, check=True)
    packages = ('duetserve_agg_decode', 'dense_ar_slo_layer_pd', 'dense_ar_elastic_layer_pd',
                'dense_ar_dynamic_layer_pd', 'dense_ar_layerwise_pd')
    frozen = {str(p.relative_to(REPO)): digest(p) for package in packages
              for p in (REPO / 'poc_flashvep' / package).glob('*.py')}
    write(ROOT / 'state/frozen_predecessor_hashes.json', frozen)
    for subdir, names in {
        'configs': ('frozen_SLO_regimes.json', 'verified_ByteLevel_text_priming.json', 'revised_cost_model_v1.json'),
        'analysis': ('exact_phase_norm_GPU_gate.json', 'phase_mask_norm_GPU_gate.json'),
        'state': ('installed_runtime_sources.json', 'canonical_block_runtime_contract.json'),
        'data': ('discovery.json', 'qualification.json'),
    }.items():
        for name in names:
            write(ROOT / subdir / name, json.loads((SLO_PRIOR / subdir / name).read_text()))
    owner = resources()
    actual, approved = owner.processes(), owner.burners()
    assert all(os.getsid(row['pid']) == approved.get(row['gpu']) for row in actual)
    write(ROOT / 'state/environment.json', dict(time=time.time(), processes=actual, burners=approved,
        GPUs=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version', '--format=csv'], text=True),
        topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
        model_config=json.loads((MODEL / 'config.json').read_text())))
    write(ROOT / 'state/progress.json', dict(start_epoch=1794846819, deadline_epoch=1794882819,
        phase='TP2_FIRST_DIVERGENCE', status='ACTIVE', results='NOT_RUN'))
    preserve()


def bind_previous():
    """Redirect evidence paths only. Do not modify model or installed sources."""
    for module in (agg, prior):
        module.ROOT, module.BASE, module.BRANCH = ROOT, BASE, BRANCH
        module.resources, module.preserve, module.initialize = resources, preserve, initialize
    for name, module in list(sys.modules.items()):
        if name.startswith(('poc_flashvep.dense_ar_slo_layer_pd.', 'poc_flashvep.duetserve_agg_decode.')):
            if hasattr(module, 'ROOT'):
                module.ROOT = ROOT


def save(config):
    initialize()
    path = ROOT / 'configs' / (config['name'] + '.json')
    if path.exists():
        assert json.loads(path.read_text()) == config
    else:
        write(path, config)
    return config['name']


if __name__ == '__main__':
    initialize()
    print(ROOT)

"""Own a new evidence namespace without changing the user checkout or index."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..dense_ar_slo_layer_pd import common as previous

REPO = previous.REPO
ROOT = REPO / 'poc_flashvep/reports/duetserve_agg_dp2_tp2_tbt_crossover_20261116'
SOURCE = Path(__file__).parent
PRIOR = previous.ROOT
BASE = 'db300024e29e6adec845bec96f90d6efcdbd0f72'
BRANCH = 'flashvep/duetserve_agg_dp2_tp2_tbt_crossover_20261116'
PYTHON = previous.PYTHON
QUIET_LOCK = previous.QUIET_LOCK
MODEL = previous.MODEL
MODEL_REVISION = previous.MODEL_REVISION
write = previous.write
digest = previous.digest
git = previous.git
preservation = previous.preservation


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    owner.ROOT = ROOT
    owner.PRIOR = PRIOR
    owner.APPROVED_LOGS.update((ROOT / 'burner.log', PRIOR / 'burner.log',
                               previous.PREVIOUS / 'burner.log', previous.STATIC / 'burner.log'))
    return owner


def preserve():
    assert preservation() == json.loads((ROOT / 'state/preservation.json').read_text())['signatures']


def initialize():
    for name in ('state', 'configs', 'raw', 'logs', 'data', 'analysis', 'figures', 'references'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    receipt = ROOT / 'state/preservation.json'
    if receipt.exists():
        preserve()
        # The inherited CPU tests require this frozen fixture even though the
        # aggregated scheduler does not use a Layer policy.
        fixture = ROOT / 'configs/revised_cost_model_v1.json'
        if not fixture.exists():
            write(fixture, json.loads((PRIOR / 'configs/revised_cost_model_v1.json').read_text()))
        contract = ROOT / 'state/canonical_block_runtime_contract.json'
        if not contract.exists():
            write(contract, json.loads((PRIOR / 'state/canonical_block_runtime_contract.json').read_text()))
        return
    assert git('rev-parse', BASE) == BASE
    write(receipt, dict(time=time.time(), signatures=preservation(), publication_parent=BASE))
    subprocess.run(['git', 'branch', BRANCH, BASE], cwd=REPO, check=True)
    frozen = {str(p.relative_to(REPO)): digest(p)
              for package in ('dense_ar_slo_layer_pd', 'dense_ar_elastic_layer_pd',
                              'dense_ar_dynamic_layer_pd', 'dense_ar_layerwise_pd')
              for p in (REPO / 'poc_flashvep' / package).glob('*.py')}
    names = ('dense_ar_slo_throughput_dynamic_layer_pd_final_summary.md',
             'dense_ar_slo_specific_colocated_frontier.md',
             'dense_ar_continuous_colocated_correctness_and_baseline.md',
             'dense_ar_four_method_qps_sweep_results.md')
    for name in names:
        path = REPO / 'poc_flashvep/reports' / name
        frozen[str(path.relative_to(REPO))] = digest(path)
    write(ROOT / 'state/frozen_predecessor_hashes.json', frozen)
    for subdir, names in {
        'configs': ('frozen_SLO_regimes.json', 'verified_ByteLevel_text_priming.json',
                    'revised_cost_model_v1.json'),
        'analysis': ('exact_phase_norm_GPU_gate.json', 'phase_mask_norm_GPU_gate.json'),
        'state': ('installed_runtime_sources.json', 'canonical_block_runtime_contract.json'),
        'data': ('discovery.json', 'qualification.json'),
    }.items():
        for name in names:
            write(ROOT / subdir / name, json.loads((PRIOR / subdir / name).read_text()))
    owner = resources()
    actual, approved = owner.processes(), owner.burners()
    assert all(os.getsid(row['pid']) == approved.get(row['gpu']) for row in actual)
    write(ROOT / 'state/environment.json', dict(time=time.time(), processes=actual, burners=approved,
        GPU_query=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version', '--format=csv'], text=True),
        topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
        model_config=json.loads((MODEL / 'config.json').read_text())))
    write(ROOT / 'state/progress.json', dict(start_epoch=time.time(), phase='DP2_QUALIFICATION',
        status='ACTIVE', experimental_results='NOT_RUN',
        stop_rule='Qualify each execution family. Preserve failures. Do not weaken an arm to obtain 2x.'))
    preserve()


def save(config):
    initialize()
    path = ROOT / 'configs' / (config['name'] + '.json')
    if path.exists():
        assert json.loads(path.read_text()) == config, 'Immutable configuration collision'
    else:
        write(path, config)
    return config['name']


def bind_previous():
    """Redirect only evidence ownership; leave the qualified model code unchanged."""
    previous.ROOT = ROOT
    previous.BASE = BASE
    previous.BRANCH = BRANCH
    previous.resources = resources
    previous.preserve = preserve
    previous.initialize = initialize
    # Some analysis modules copy ROOT at import time. A prior correctness
    # helper can import them before this adapter. Redirect already loaded
    # evidence modules too. Do not change their model source or SOURCE path.
    for name, module in list(sys.modules.items()):
        if name.startswith('poc_flashvep.dense_ar_slo_layer_pd.') and hasattr(module, 'ROOT'):
            module.ROOT = ROOT


if __name__ == '__main__':
    initialize()
    print(ROOT)

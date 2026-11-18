"""Keep user state and every predecessor artifact unchanged."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from ..dense_ar_layer_token_pd import common as predecessor
from ..dense_ar_slo_layer_pd import common as foundation

REPO = foundation.REPO
SOURCE = Path(__file__).parent
ROOT = REPO / 'poc_flashvep/reports/dense_ar_matched_axis_crossover_20261117'
PREVIOUS = predecessor.ROOT
BASE = '228a041f35cedcee9f45beba6c462b453fb3a2c5'
BRANCH = 'flashvep/dense-ar-matched-axis-crossover-20261117'
PYTHON, MODEL, MODEL_REVISION = foundation.PYTHON, foundation.MODEL, foundation.MODEL_REVISION
QUIET_LOCK = foundation.QUIET_LOCK
write, digest, git, preservation = foundation.write, foundation.digest, foundation.git, foundation.preservation
PACKAGES = ('dense_ar_layer_token_pd', 'duetserve_agg_decode', 'dense_ar_slo_layer_pd',
            'dense_ar_elastic_layer_pd', 'dense_ar_dynamic_layer_pd', 'dense_ar_layerwise_pd')


def preserve():
    assert preservation() == json.loads((ROOT/'state/preservation.json').read_text())['signatures'], 'User state changed'


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    owner.ROOT, owner.PRIOR = ROOT, ROOT/'no_legacy_manifest_scan'
    owner.APPROVED_LOGS.update((ROOT/'burner.log', PREVIOUS/'burner.log',
        foundation.ROOT/'burner.log', foundation.PREVIOUS/'burner.log',
        foundation.STATIC/'burner.log', predecessor.PREVIOUS/'burner.log'))
    for path in (PREVIOUS/'state').glob('burner_*.json'):
        value = json.loads(path.read_text())
        if isinstance(value, dict) and 'GPUs' in value and 'supervisor' in value:
            target = ROOT/'state'/('burner_inherited_'+str(value['supervisor'])+'.json')
            if not target.exists():
                write(target, value)
    return owner


def initialize():
    for name in ('state', 'configs', 'raw', 'logs', 'data', 'analysis', 'figures', 'references'):
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    receipt = ROOT/'state/preservation.json'
    if receipt.exists():
        fixture=ROOT/'configs/revised_cost_model_v1.json'
        if not fixture.exists():
            write(fixture,json.loads((PREVIOUS/'configs/revised_cost_model_v1.json').read_text()))
        preserve()
        return
    assert git('rev-parse', BASE) == BASE
    assert not git('branch', '--list', BRANCH), 'Do not reuse another campaign branch'
    write(receipt, dict(time=time.time(), signatures=preservation(), publication_parent=BASE))
    subprocess.run(['git', 'branch', BRANCH, BASE], cwd=REPO, check=True)
    frozen = {str(p.relative_to(REPO)): digest(p) for name in PACKAGES
              for p in (REPO/'poc_flashvep'/name).glob('*.py')}
    for name in ('dense_ar_layer_token_dynamic_headroom_final_summary.md',
                 'dense_ar_layer_vs_token_online_results.md',
                 'dense_ar_layer_vs_token_correctness_and_execution.md',
                 'duetserve_agg_dp2_tp2_tbt_crossover_final_summary.md'):
        p = REPO/'poc_flashvep/reports'/name
        frozen[str(p.relative_to(REPO))] = digest(p)
    write(ROOT/'state/frozen_predecessor_hashes.json', frozen)
    for subdir, names in {
        'configs': ('frozen_SLO_regimes.json', 'verified_ByteLevel_text_priming.json', 'revised_cost_model_v1.json'),
        'analysis': ('exact_phase_norm_GPU_gate.json', 'phase_mask_norm_GPU_gate.json'),
        'state': ('installed_runtime_sources.json', 'canonical_block_runtime_contract.json'),
        'data': ('discovery.json', 'qualification.json', 'heldout128_8000.json'),
    }.items():
        for name in names:
            write(ROOT/subdir/name, json.loads((PREVIOUS/subdir/name).read_text()))
    owner = resources()
    approved, actual = owner.burners(), owner.processes()
    assert all(os.getsid(r['pid']) == approved.get(r['gpu']) for r in actual), 'Unknown GPU owner remains untouched'
    write(ROOT/'state/environment.json', dict(time=time.time(), GPUs=subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version', '--format=csv'], text=True),
        topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
        approved_burners=approved, processes=actual, model_revision=MODEL_REVISION,
        model_config=json.loads((MODEL/'config.json').read_text())))
    write(ROOT/'state/progress.json', dict(start_epoch=time.time(), status='ACTIVE',
        phase='UNIFIED_WORKER_AND_STATE_QUALIFICATION', primary_admission_cap=8,
        scope='No dynamic policy, TP2 repair or best-vs-best search. Preserve failed runs.'))
    write(ROOT/'state/design_freeze_v1.json', dict(time=time.time(), B=8192, S=1024,
        PD_common_admission_cap=8, source_macro_limit=1,
        admission_policy='FIFO, no active D cohort, same per-iteration query-token budget and request cap in all PD methods',
        batch_scope='Actual D cohort is bounded by the sum of original prompt query counts. This count is8000 for a layer suffix, token suffix length for a token continuation, and1 for a cached Native first query. Shapes are not forced to match.',
        source_scope='Every P prefix exceeds4096 query rows. Thus at most one complete prompt prefix fits B8192. The coordinator source macro is1. Worker S is still1024.',
        primary_requests=144, qualification_requests=[8,32,128],
        arithmetic_scope='N128 cannot exactly match integer suffix rows for all three cuts. N144 supports exact16-token-aligned plans.',
        frozen_SLO_SHA256=digest(ROOT/'configs/frozen_SLO_regimes.json'),
        evidence='STRUCTURAL_ACCOUNTING_NOT_PERFORMANCE'))
    preserve()


def bind_legacy():
    """Bind evidence globals before importing frozen execution modules."""
    from ..duetserve_agg_decode import common as agg
    for module in (foundation, predecessor, agg):
        module.ROOT, module.BASE, module.BRANCH = ROOT, BASE, BRANCH
        module.initialize, module.preserve, module.resources = initialize, preserve, resources
    for name, module in list(sys.modules.items()):
        if name.startswith(('poc_flashvep.dense_ar_slo_layer_pd.',
                            'poc_flashvep.dense_ar_layer_token_pd.',
                            'poc_flashvep.duetserve_agg_decode.')) and hasattr(module, 'ROOT'):
            module.ROOT = ROOT


def save(cfg):
    initialize()
    path = ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():
        assert json.loads(path.read_text()) == cfg, 'Immutable configuration changed'
    else:
        write(path, cfg)
    return cfg['name']


if __name__ == '__main__':
    initialize()
    print(ROOT)

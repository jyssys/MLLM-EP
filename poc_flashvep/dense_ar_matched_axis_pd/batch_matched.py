"""Compare fixed axes with actual single-request batches on both GPUs."""
import argparse
import collections
import json
import time
from .common import ROOT, save, write, digest, preserve
from .configuration import base, matched_token_plan
from .prepare import qualification
from .qualification import gate
from .campaign import run_job
from .run import execute
from .host_paths import analyze

DESIGN = ROOT/'state/batch_matched_design_20261118.json'
ARMS = (('Native', 0),)+tuple((a, m) for m in (4, 8, 12) for a in ('Layer', 'Token'))
CONFIRM_SEEDS = (2026111820, 2026111821, 2026111822)


def enforce(cfg):
    cfg.update(P_group_size=1, D_admission_batch=1,
               common_prospective_KV_reservation=True, execution_contract_version=3,
               batch_matched_control=True, matched_request_batch=1,
               source_outstanding_limit=32)
    if cfg.get('token_axis'):
        cfg['token_suffix_by_request'] = matched_token_plan(cfg['static_layers'], cfg['count'], group_size=1)
    assert cfg['P_slots'] == cfg['D_slots'] == 1024
    assert cfg['token_budget'] == cfg['D_token_budget'] == 8192
    return cfg


def state_configuration(method, layers, count=18):
    cfg = qualification(method, count, layers or 12, version=20261118)
    cfg.update(name=f'batch1_state_{method}_Last{layers}_N{count}_v1',
               matrix_stage='batch1_state', phase='ACTUAL_BATCH_ONE_FULL_STATE_GATE')
    return enforce(cfg)


def performance_configuration(method, layers, cap, qps, seed, stage, count):
    name = f'{stage}_{method}_Last{layers}_N{count}_O{cap}_q{qps:g}_s{seed}_P1D1_v1'
    cfg = base(name, method, layers or 12, count, cap, qps, seed, 'matched144.json',
               phase='ACTUAL_BATCH_ONE_'+stage.upper(), D_admission_batch=1,
               reference_name='HISTORICAL_original_outputs', matrix_stage=stage)
    if method == 'Native':
        cfg['static_layers'] = 0
    return enforce(cfg)


def matrix():
    rows = []
    # Freeze a small common grid before looking at any new performance.
    for index, (cap, qps) in enumerate([(16, q) for q in (2., 6., 10., 4., 8.)]
                                      + [(64, q) for q in (2., 6., 10.)]):
        ordered = ARMS[index % len(ARMS):]+ARMS[:index % len(ARMS)]
        for method, layers in ordered:
            rows.append(performance_configuration(method, layers, cap, qps,
                        2026111810, 'batch1_discovery', 72))
    for rep, seed in enumerate(CONFIRM_SEEDS):
        for index, cap in enumerate((16, 64)):
            shift = (rep+index) % len(ARMS)
            for method, layers in ARMS[shift:]+ARMS[:shift]:
                rows.append(performance_configuration(method, layers, cap, 6., seed,
                            'batch1_confirmation', 144))
    return rows


def freeze():
    configs = matrix()
    design = dict(protocol='ACTUAL_P1_D_PREPARATION1_DECODE1', B=8192, S=1024,
        layers=[4, 8, 12], outputs=[16, 64], discovery_count=72,
        discovery_seed=2026111810, confirmation_count=144,
        confirmation_seeds=list(CONFIRM_SEEDS), confirmation_QPS=6.,
        actual_batch_required=dict(P_forward_requests=1, D_prefill_requests=1, D_decode_requests=1),
        fixed_SLO_SHA256=digest(ROOT/'configs/frozen_SLO_regimes.json'),
        common_SLO_SHA256=digest(ROOT/'configs/common_output_sweep_SLO.json'),
        names=[c['name'] for c in configs], total_starts=len(configs),
        scope='Batch-matched causal control. Original BF16, model, prompt pool, numerical correction, NIXL contract, worker B/S and SLO stay fixed. Keep waiting, collection and decode residency in client metrics. Do not call this an optimized unrestricted Token/Layer baseline. Do not infer a crossover from isolated GEMM or a single start. Last2 full sweep stays cancelled. Batch2/3 is not implemented by this protocol.',
        reporting='Keep original cap8 evidence separate. All costs include the actual online path. Label finite cohorts, host enclosures and isolated GPU diagnostics separately.')
    if DESIGN.exists():
        assert {k:v for k,v in json.loads(DESIGN.read_text()).items() if k!='frozen_epoch'} == design
    else:
        write(DESIGN, dict(frozen_epoch=time.time(), **design))
    for cfg in configs:
        save(cfg)
    return configs


def audit(name):
    folder = ROOT/'raw'/name
    cfg = json.loads((folder/'configuration.json').read_text())
    assert cfg['batch_matched_control'] and cfg['P_group_size'] == cfg['D_admission_batch'] == 1
    observed = {}
    for role in ('prefill', 'decode'):
        shapes = json.loads((folder/f'measured_geometry_{role}.json').read_text())['counts']
        counts = collections.Counter()
        for shape, count in shapes.items():
            value = json.loads(shape)
            assert value['requests'] == 1, (name, role, value)
            counts['prefill' if value['prefilling'] else 'decode'] += count
            if not value['prefilling']:
                assert value['query_tokens'] == value['padded_query_tokens'] == 1, (name, value)
        observed[role] = dict(counts)
    cfg_metrics = json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
    for field in ('source_groups', 'destination_groups'):
        assert len(cfg_metrics[field]) == cfg['count'] and all(len(g)==1 for g in cfg_metrics[field]), (name, field)
    assert cfg_metrics['output_gate']['passed'] and cfg_metrics['actual_query_coverage_gate']['passed']
    result = dict(name=name, passed=True, observed=observed,
                  actual_P_request_batch=1, actual_D_prefill_request_batch=1,
                  actual_D_decode_request_batch=1,
                  worker_BS_passed=cfg_metrics['unified_worker_gate'],
                  evidence='TRACE_ANALYSIS', scope='Actual prepared input shapes, not a configuration-only claim. Batch1 has no graph-padding difference. Native has no remote prompt-prefill query.')
    write(ROOT/'analysis'/(name+'_matched_batch_gate.json'), result)
    return result


def qualify():
    marker = ROOT/'state/batch1_full_state_qualified.json'
    if marker.exists():
        assert json.loads(marker.read_text())['passed']
        return
    values = []
    reference = state_configuration('Native', 0)
    for method, layers in ARMS:
        cfg = state_configuration(method, layers)
        assert execute(save(cfg)), cfg['name']
        values.append(gate(cfg['name'], reference['name']))
    write(marker, dict(epoch=time.time(), passed=True, count=18,
                       names=[v['name'] for v in values], evidence='CORRECTNESS_ONLY'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    preserve()
    configs = freeze()
    if args.prepare:
        print('BATCH1_FROZEN', len(configs), flush=True)
        return
    assert json.loads((ROOT/'state/user_batch_matched_scope_revision_20261118.json').read_text())['status']=='TRANSITION_COMPLETE'
    try:
        qualify()
        for index, cfg in enumerate(configs):
            run_job(cfg)
            audit(cfg['name'])
            analyze(cfg['name'])
            write(ROOT/'state/batch1_progress.json', dict(epoch=time.time(), status='ACTIVE',
                completed=index+1, total=len(configs), last=cfg['name']))
            print('BATCH1_PROGRESS', index+1, len(configs), cfg['name'], flush=True)
        write(ROOT/'state/batch1_complete.json', dict(epoch=time.time(), count=len(configs),
             scope='Required actual single-request batch controls completed. Completion does not establish Layer advantage or a crossover.'))
    except BaseException as error:
        write(ROOT/'state/batch1_stopped.json', dict(epoch=time.time(), error=repr(error)))
        raise


if __name__ == '__main__':
    main()

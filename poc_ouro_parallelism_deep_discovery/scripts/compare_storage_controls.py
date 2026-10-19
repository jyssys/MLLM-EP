"""Compare full-core storage controls. No causal cache-hit attribution."""
import argparse
import datetime as dt
import statistics
import numpy as np
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.scripts.native_parity_analysis import stats, log_softmax


def complete_run(size, P, arm, suffix):
    short = size.replace('.', '').replace('B', '')
    root = ROOT / 'outputs' / f'storage_{arm}_base{short}_P{P}_{suffix}'
    status = root / 'guard_status.json'
    if not status.exists() or not read(status).get('success'):
        return None
    return root / f'{size}_R4_TP{P}'


def trials(path):
    return [read(f) for f in sorted(path.glob('P*_request*_repeat*.json'))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suffix',default='v1')
    args = parser.parse_args()
    assert not args.output.exists()
    records = []
    for size in MODELS:
        for P in [2, 4]:
            alias, copies = [complete_run(size, P, arm,args.suffix) for arm in ['alias', 'copies']]
            if alias is None or copies is None:
                continue
            left, right = trials(alias), trials(copies)
            assert len(left) == len(right) == 30
            config_a, config_b = [read(path / 'engine_config.json') for path in [alias, copies]]
            caches=[config['compilation_config'].pop('cache_dir',None) for config in [config_a,config_b]]
            if caches[0] is not None or caches[1] is not None:
                assert all(caches) and caches[0]!=caches[1]
            assert config_a == config_b
            observations = []
            for N in [1024, 4096, 8192]:
                a = [v for v in left if v['prompt'] == N]
                b = [v for v in right if v['prompt'] == N]
                assert len(a) == len(b) == 10
                median_a = statistics.median(v['wall_ms'] for v in a)
                median_b = statistics.median(v['wall_ms'] for v in b)
                paired = []
                for request_id in sorted({v['request_id'] for v in a}):
                    av = statistics.median(v['wall_ms'] for v in a if v['request_id'] == request_id)
                    bv = statistics.median(v['wall_ms'] for v in b if v['request_id'] == request_id)
                    paired.append(dict(request_id=request_id, alias_ms=av, copies_ms=bv, copies_over_alias=bv/av))
                observations.append(dict(context=N, alias_median_E2E_ms=median_a, copies_median_E2E_ms=median_b,
                                         copies_over_alias_speedup=median_b/median_a, paired_requests=paired,
                                         N_trials_each=10, N_distinct_request_shapes=2))
            ae, be = [read(path / 'parity_capture.json')['evidence'] for path in [alias, copies]]
            captures = []
            for rank in range(P):
                a = next(v for v in ae if v['rank'] == rank)
                b = next(v for v in be if v['rank'] == rank)
                assert a['positions'] == b['positions'] and a['KV_names'] == b['KV_names']
                av, bv = np.load(a['path']), np.load(b['path'])
                lp, lq = log_softmax(av['logits']), log_softmax(bv['logits'])
                captures.append(dict(rank=rank, logits=stats(bv['logits'], av['logits']),
                                     hidden=stats(bv['hidden'], av['hidden']), KV=stats(bv['KV'], av['KV']),
                                     top1_match=(av['logits'].argmax(-1) == bv['logits'].argmax(-1)).tolist(),
                                     KL_alias_to_copies=(np.exp(lp)*(lp-lq)).sum(-1).tolist()))
            storage = {}
            for arm, path in [('alias', alias), ('copies', copies)]:
                raw = read(path / 'storage_control.json')
                storage[arm] = [{k:v for k,v in row.items() if k != 'explicitly_initialized_copy_names'} for row in raw]
            records.append(dict(model=size, R=4, P=P, alias_path=str(alias), copies_path=str(copies),
                                compile_cache_paths=caches,vllm_index_cache_isolated=all(caches) and caches[0]!=caches[1],
                                global_cache_environment_records=[read(path/'cache_environment.json')
                                    if (path/'cache_environment.json').exists() else None for path in [alias,copies]],
                                compiled_storage_execution='Requires separate generated-graph operand validation. Parameter allocation and numerical equality alone do not establish execution ownership.',
                                timing=observations, capture_comparisons=captures, storage=storage,
                                evidence_class='LIVE_E2E', quality='Four controlled teacher-forced positions only. Natural accuracy NOT_RUN.',
                                attribution='Physical sharing changes parameter/module aliases and may change compilation. HBM counters unavailable.',
                                uncertainty='Sequential arm order. Two request shapes do not establish broad request-level confidence.'))
    write(args.output, dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(), records=records,
                           candidate='NONE', scope='Full-core alias/copies counterfactual. No new parallel architecture or hardware-cache claim.'))
    for row in records:
        print(row['model'], row['P'], [(v['context'], round(v['copies_over_alias_speedup'],4)) for v in row['timing']],
              'logit_equal', [v['logits']['bitwise_float_capture_equal'] for v in row['capture_comparisons']])


if __name__ == '__main__':
    main()

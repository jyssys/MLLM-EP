"""A conditional cache-capacity bound. No CUDA call or measured traffic claim."""
import argparse
from functools import lru_cache
import itertools
from poc_ouro_parallelism_deep_discovery.contract import *


def minimum_fetches(items, passes, capacity, warm=False):
    sequence = tuple(range(items)) * passes
    @lru_cache(None)
    def solve(index, cache):
        if index == len(sequence):
            return 0
        value = sequence[index]
        resident = frozenset(cache)
        if value in resident:
            return solve(index + 1, cache)
        # Grant free bypass. This favors persistent residency and is not an H100 policy claim.
        choices = [solve(index + 1, cache)]
        if capacity:
            if len(resident) < capacity:
                choices.append(solve(index + 1, tuple(sorted(resident | {value}))))
            else:
                for evicted in resident:
                    choices.append(solve(index + 1, tuple(sorted((resident - {evicted}) | {value}))))
        return 1 + min(choices)
    if not warm:
        return solve(0, ())
    return min(solve(0, tuple(cache)) for cache in itertools.combinations(range(items), min(capacity, items)))


def bounds(working_set, capacity, passes):
    if min(working_set, capacity, passes) < 0 or working_set == 0 or passes == 0:
        raise ValueError('Require a positive working set and pass count, with nonnegative capacity')
    retained = min(working_set, capacity)
    no_cache = passes * working_set
    cold_min = no_cache - (passes - 1) * retained
    warm_min = passes * (working_set - retained)
    return dict(matrix_working_set_bytes_per_rank=working_set, cache_capacity_bytes=capacity,
                passes=passes, no_cache_one_fetch_per_pass_bytes=no_cache,
                cold_first_pass_minimum_fetch_bytes=cold_min,
                warm_initial_cache_minimum_fetch_bytes=warm_min,
                cold_maximum_fetch_reduction_fraction=(no_cache - cold_min) / no_cache,
                warm_maximum_fetch_reduction_fraction=(no_cache - warm_min) / no_cache)


def exhaustive_check():
    count = 0
    for items in range(1, 7):
        for capacity in range(0, min(items, 3) + 1):
            for passes in range(1, 5):
                result = bounds(items, capacity, passes)
                for warm, key in [(False, 'cold_first_pass_minimum_fetch_bytes'),
                                  (True, 'warm_initial_cache_minimum_fetch_bytes')]:
                    actual = minimum_fetches(items, passes, capacity, warm)
                    assert actual == result[key], (items, capacity, passes, warm, actual, result[key])
                    count += 1
    return dict(checked_optimal_toy_cases=count, status='PASS',
                scope='Exact finite cache search with bypass. Not a GPU cache-policy simulation.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records, configs = [], []
    for size, model in MODELS.items():
        config_file = model['path'] / 'config.json'
        config = read(config_file)
        h, intermediate, layers = config['hidden_size'], config['intermediate_size'], config['num_hidden_layers']
        assert (h, intermediate, layers) == (2048, 5632, model['layers'])
        per_layer = 2 * (4 * h * h + 3 * h * intermediate)
        configs.append(dict(size=size, path=str(config_file), sha256=sha(config_file),
                            layers=layers, hidden_size=h, intermediate_size=intermediate,
                            BF16_matrix_bytes_per_physical_layer=per_layer))
        for tp in (1, 2, 4, 8):
            assert per_layer * layers % tp == 0
            working_set = per_layer * layers // tp
            for depth in (2, 3, 4):
                records.append(dict(size=size, tp=tp, executed_passes=depth,
                                    hardware_scope='P8_PROJECTION' if tp == 8 else 'CAPACITY_SCREEN_ONLY',
                                    **bounds(working_set, 50 * 1024 ** 2, depth)))
    write(args.output, dict(evidence_class='ANALYTIC_BOUND', configs=configs,
        validation=exhaustive_check(), records=records,
        hardware_source='artifacts/persistent_execution_audit.json: measured L2 on physical4 and6 is52,428,800 bytes. Use50MiB uniformly as the stated H100-family capacity assumption.',
        assumptions=['One required fetch per distinct BF16 linear-weight byte in each complete pass.',
                     'Identical physical matrix working set and fixed access order in every pass.',
                     'All nominal L2 capacity is available for these weights, with free bypass and optimal replacement.',
                     'No prefetch transaction is free. A fetch is counted whenever bytes enter from outside this cache.',
                     'Cold and optimally preloaded initial-cache views are separate.',
                     'No persistent register/shared-memory residency, compression, extra ready rows, or operator reordering.'],
        limitations=['Not actual HBM traffic or latency.', 'Not a measured alias/copies timing bound.',
                     'No E2E speedup follows from a fetch-volume fraction.',
                     'Does not bound compiler, instruction-cache, capacity, or other ownership mechanisms.',
                     'Adaptive request depths select a mixture of these structural cases, not less allocated KV.']))
    print(json.dumps(dict(records=len(records), validation=exhaustive_check())))


if __name__ == '__main__':
    main()

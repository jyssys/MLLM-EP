"""CPU equivalence tests against the native written-position implementation."""
import copy
import fcntl
import random
import statistics
import sys
import time
from .common import ROOT, RLT, write
from .bulk_written import add_span


def main():
    sys.path.insert(0, str(RLT))
    from vllm_rlt.core.kv_cache_manager import _WrittenPositions
    rng = random.Random(2026110359)
    for case in range(20000):
        state = _WrittenPositions()
        for position in rng.sample(range(256), rng.randrange(256)):
            state.add(position)
        expected = copy.deepcopy(state)
        actual = copy.deepcopy(state)
        start = rng.randrange(300)
        end = rng.randrange(start, 350)
        for position in range(start, end):
            expected.add(position)
        add_span(actual, start, end)
        assert (actual.prefix, actual.pending) == (expected.prefix, expected.pending), case
    samples = []
    for label, layers, loops, rows in [('ouro', 24, 4, 4096), ('huginn', 8, 32, 2560)]:
        for repeat in range(5):
            baseline = [[_WrittenPositions() for _ in range(layers)] for _ in range(loops)]
            candidate = copy.deepcopy(baseline)
            begin = time.perf_counter()
            for depth in range(loops):
                for position in range(rows):
                    for layer in range(layers):
                        baseline[depth][layer].add(position)
            original_s = time.perf_counter()-begin
            begin = time.perf_counter()
            for depth in range(loops):
                for layer in range(layers):
                    add_span(candidate[depth][layer], 0, rows)
            bulk_s = time.perf_counter()-begin
            assert baseline == candidate
            samples.append(dict(model_geometry=label, repeat=repeat, rows=rows, layers=layers, loops=loops,
                                original_host_seconds=original_s, bulk_host_seconds=bulk_s,
                                scope='CPU bookkeeping microbenchmark. Excludes row grouping and all GPU execution. Not an E2E speedup.'))
    result = dict(seed=2026110359, randomized_native_equivalence_cases=20000,
                  all_exact=True, samples=samples,
                  scope='Metadata tests only. Model output and GPU KV identity require separate native execution.')
    write(ROOT/'state/bulk_written_cpu_validation.json', result)
    print('PASS: 20000 native-state comparisons and 10 exact CPU controls')


if __name__ == '__main__':
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()

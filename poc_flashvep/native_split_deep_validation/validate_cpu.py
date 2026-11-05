"""Retain final CPU regressions and source hashes without using a GPU."""
import io
import unittest
from .common import *


def main():
    initialize()
    prior.parent.prior.preserve()
    output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(SOURCE), pattern='test_*.py', top_level_dir=str(REPO))
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    receipt = dict(time=time.time(), passed=result.wasSuccessful(), tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        output=output.getvalue(), source_SHA256={str(path.relative_to(SOURCE)): digest(path)
            for path in sorted(SOURCE.glob('*.py'))},
        scope='CPU regression only. This receipt does not replace GPU output, KV, hidden-state, or work qualification.')
    write(ROOT / 'state/final_CPU_regression.json', receipt)
    print(receipt['output'], end='', flush=True)
    prior.parent.prior.preserve()
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()

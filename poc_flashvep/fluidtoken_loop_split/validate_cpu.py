"""Run explicit CPU regression modules outside clean GPU intervals."""
import io
import time
import unittest
from .common import ROOT, SOURCE, configure, digest, write


def main():
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('final_explicit_CPU_regressions'):
        paths = sorted(SOURCE.glob('test_*.py'))
        names = [__package__ + '.' + path.stem for path in paths]
        suite = unittest.defaultTestLoader.loadTestsFromNames(names)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        record = dict(time=time.time(), command='python -m ' + __package__ + '.validate_cpu',
                      modules=names, tests=result.testsRun, passed=result.wasSuccessful(),
                      failures=len(result.failures), errors=len(result.errors),
                      output=stream.getvalue(), source_SHA256={str(p): digest(p) for p in paths},
                      scope='CPU regressions only. These tests do not qualify GPU performance or model fidelity.')
        target = ROOT / 'state' / ('cpu_validation_' + str(int(record['time'])) + '.json')
        write(target, record)
        print(record['output'], flush=True)
        print(str(target), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Run evidence tests on CPU and preserve the result."""
import io
import platform
import sys
import time
import unittest
from .contract import ROOT,write


def main():
    suite=unittest.defaultTestLoader.loadTestsFromName('poc_flashvep.ouro_tp4.test_evidence')
    output=io.StringIO();start=time.perf_counter()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    write(ROOT/'tests.json',dict(scope='CPU evidence and provenance tests. Physical tests are separate raw artifacts.',
        interpreter=sys.executable,python=platform.python_version(),tests_run=result.testsRun,
        successful=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),wall_s=time.perf_counter()-start,output=output.getvalue()))
    print(output.getvalue(),end='')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()

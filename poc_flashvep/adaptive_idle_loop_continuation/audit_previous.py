"""Replay the previous local-tail gate before changing instrumentation."""
import csv
from .common import *


def main():
    initialize()
    rows=list(csv.DictReader((PREVIOUS/'instrumentation_qualification.csv').open()))
    retained=[v for v in rows if v['collector']=='thin_trace128']
    result=dict(previous=[dict(v) for v in retained],passed=all(v['qualified']=='True' for v in retained),
        scope='Frozen trace replay. A red result reproduces the missing qualified-slack evidence, not a model correctness failure.')
    write(ROOT/'state/previous_instrumentation_red_replay.json',result)
    print(json.dumps(result))
    assert result['passed'],'Previous thin traces fail the 5% local-tail gate'


if __name__=='__main__':main()

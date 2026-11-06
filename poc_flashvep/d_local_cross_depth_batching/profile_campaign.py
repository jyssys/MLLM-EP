"""Run the primed collector only after its GPU self-test passes."""
import time
from .common import *
from .run import execute
from .profile_analysis import analyze


def main():
    test=ROOT/'raw/thin_collector_selftest/thin_selftest.json'
    while not test.exists():
        assert time.time()<LONG_START_CUTOFF
        time.sleep(5)
    assert json.loads(test.read_text())['passed']
    for model in ('ouro','huginn'):
        name=f'local_{model}_native_thin_trace128'
        if execute(name):
            result=analyze(name,f'local_{model}_native_n128_s2026110701_off')
            write(ROOT/'state'/f'{model}_thin_profile_gate.json',result)


if __name__=='__main__':main()

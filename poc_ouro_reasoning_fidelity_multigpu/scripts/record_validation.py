"""Keep implementation checks separate from a falsified scientific assumption."""
import os
import subprocess
import argparse
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def execute(modules):
    result = subprocess.run([CPU, '-m', 'unittest', *modules, '-v'], cwd=PROJECT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return dict(exit_code=result.returncode, modules=modules, output=result.stdout)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/validation_execution.json')
    options=parser.parse_args()
    prefix = 'poc_ouro_reasoning_fidelity_multigpu.tests.'
    implementation = execute([prefix + name for name in
        ['test_contract', 'test_logit_metrics', 'test_problem_clusters', 'test_guard_scope']])
    assumption = execute([prefix + 'test_restart_property'])
    target = (ROOT/options.output).resolve()
    assert target.is_relative_to(ROOT/'results')
    assert not target.exists(), 'Preserve earlier validation results.'
    write(target, dict(implementation=implementation, scientific_assumption=assumption,
        all_tests_pass=implementation['exit_code'] == 0 and assumption['exit_code'] == 0,
        interpretation='The historical failure remains a failed property. Its two runs have different preceding request histories.',
        limitation='Equal engine kwargs and request seeds alone do not control the complete execution history.',
        benchmark_grades_changed=False, GPU_execution=False))
    assert implementation['exit_code'] == 0
    assert assumption['exit_code'] != 0, 'Review an unexpected change in the historical evidence.'
    print('Implementation checks pass. The historical scientific property still fails.')


if __name__ == '__main__':
    main()

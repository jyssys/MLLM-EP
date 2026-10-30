"""Parallel correctness only. Clean serving remains isolated."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import time

from .common import ROOT, REPO, setup, write
from .jobs import configuration, run, parity
from . import resources


def main():
    setup()
    write(ROOT / 'state/qualification_controller.json', dict(pid=os.getpid(), status='RUNNING', time=time.time()))
    resources.stop()
    try:
        jobs = [configuration(model, arm, 'correctness', 40, 0, 20261029,
                              gpus=(pair, pair + 1), capacity=1, cap=32, diagnostic=True,
                              name=f'{model}_qual_{arm}')
                for model, arm, pair in [('ouro', 'PD_G1', 0), ('ouro', 'B', 2),
                                          ('huginn', 'PD_G1', 4), ('huginn', 'B', 6)]]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(run, jobs))
        result = {model: parity(f'{model}_qual_PD_G1', f'{model}_qual_B') for model in ('ouro', 'huginn')}
        for model in result:
            result[model]['passed'] = result[model]['N'] == 40 and result[model]['exact'] == 40 and result[model]['input_equal'] == 40
        write(ROOT / 'configs/qualification.json', result)
        heavy = [configuration(model, 'B', 'correctness', 2, 0, 20261029,
                               gpus=(pair, pair + 1), capacity=1, cap=8, diagnostic=True,
                               env=dict(ABC_VERIFY_KV='1', ABC_REVERSE_FRAGMENTS='1'), name=f'{model}_KV_identity_B')
                 for model, pair in [('ouro', 0), ('huginn', 4)] if result[model]['passed']]
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(run, heavy))
        reference = []
        for model, pair in [('ouro', 0), ('huginn', 4)]:
            reference.append(configuration(model, 'Native', 'correctness', 8, 0, 20261029,
                                           gpus=(pair,), capacity=1, cap=32, diagnostic=True, name=f'{model}_reference_RLT'))
            reference.append(configuration(model, 'HF', 'correctness', 8, 0, 20261029,
                                           gpus=(pair + 1,), capacity=1, cap=32, diagnostic=True,
                                           reference=True, python='/home/esjung/.venvs/looped-pd-abc-hf457/bin/python',
                                           name=f'{model}_reference_HF'))
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(run, reference))
        references = {model: parity(f'{model}_reference_RLT', f'{model}_reference_HF') for model in ('ouro', 'huginn')}
        write(ROOT / 'configs/reference_comparison.json', references)
        write(ROOT / 'state/qualification_controller.json', dict(pid=os.getpid(), status='COMPLETE_REVIEW_REQUIRED', time=time.time()))
    finally:
        resources.restore()


if __name__ == '__main__':
    main()

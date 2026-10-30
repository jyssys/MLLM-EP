"""Small native CUDA profiles. Never merge these times with clean serving."""
import time

from .common import ROOT, REPO, remaining, write
from .jobs import configuration, run


def main():
    attempts = []
    for model in ('ouro', 'huginn'):
        for arm in ('PD_G1', 'B'):
            if remaining() < 1200:
                write(ROOT / 'configs/profile_attempts.json', dict(attempts=attempts, status='TIME_BUDGET_LIMIT'))
                return
            name = f'{model}_profile_{arm}'
            job = configuration(model, arm, 'diagnostic', 2, 0, 2026108001,
                                dataset='W2', cap=4, capacity=1, diagnostic=True,
                                python=str(REPO / 'poc_flashvep/looped_pd_b_slo_serving/profile_python.py'),
                                name=name, timeout=600, env=dict(B_EVENT_TRACE='1'))
            status = run(job)
            attempts.append(dict(run=name, status=status, time=time.time(),
                                 evidence='HEAVY_PROFILE_DIAGNOSTIC_NOT_CLEAN_SERVING'))
            write(ROOT / 'configs/profile_attempts.json', dict(attempts=attempts))
            if status['status'] != 'COMPLETE':
                # Do not spend the whole remaining allowance on a profiler compatibility failure.
                return


if __name__ == '__main__':
    main()

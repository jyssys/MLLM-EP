"""Run a bounded sequence of fixed-depth generic baseline jobs with the owned guard."""
import datetime as dt
import os
import subprocess
import sys
from poc_ouro_parallelism_deep_discovery.contract import *


def main():
    plan = [('1.4B', 1, [4]), ('2.6B', 1, [4]), ('2.6B', 2, [4, 6]),
            ('1.4B', 4, GPUS), ('2.6B', 4, GPUS)]
    ledger = ROOT / 'artifacts/native_cohort_queue_v1.json'
    if ledger.exists():
        raise FileExistsError(ledger)
    records = []
    write(ledger, dict(status='RUNNING', jobs=records,
                       completed_before_queue='outputs/native_cohort_base14_P2_v1',
                       scope='Generic fixed-R4 baseline. Not a candidate or adaptive CDB experiment.'))
    for size, tp, gpus in plan:
        short = size.replace('.', '').replace('B', '')
        target = f'outputs/native_cohort_base{short}_P{tp}_v1'
        if (ROOT / target).exists():
            raise FileExistsError(ROOT / target)
        command = [REFERENCE_PYTHON, '-m', f'{ROOT.name}.scripts.gpu_session',
                   '--gpus', ','.join(map(str, gpus)), '--output', target, '--timeout', '2400', '--',
                   PYTHON, '-m', f'{ROOT.name}.scripts.native_cohort_control',
                   '--size', size, '--tp', str(tp), '--protocol',
                   str(ROOT / 'artifacts/native_cohort_protocol_v2.json')]
        record = dict(size=size, tp=tp, physical_gpus=gpus, command=command,
                      started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status='RUNNING')
        records.append(record)
        write(ledger, dict(status='RUNNING', jobs=records))
        result = subprocess.run(command, cwd=PROJECT, env=os.environ.copy())
        record.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      returncode=result.returncode, status='COMPLETE' if result.returncode == 0 else 'FAILED')
        write(ledger, dict(status='RUNNING' if result.returncode == 0 else 'STOPPED_ON_FAILURE', jobs=records))
        if result.returncode:
            raise SystemExit(result.returncode)
    write(ledger, dict(status='COMPLETE', jobs=records))


if __name__ == '__main__':
    main()

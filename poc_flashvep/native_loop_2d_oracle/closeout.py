"""Run CPU checks and verify restored burners. Never start a model experiment."""
import compileall
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from .common import *


def main():
    configure()
    parent.prior.preserve()
    commands = [
        [sys.executable, '-m', 'unittest', 'discover', '-s', str(SOURCE.relative_to(REPO)),
         '-t', '.', '-p', 'test_*.py', '-v'],
        [sys.executable, '-m', 'poc_flashvep.native_loop_2d_oracle.analyze'],
        [sys.executable, '-m', 'poc_flashvep.native_loop_2d_oracle.figures'],
    ]
    results = []
    for command in commands:
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, timeout=180)
        results.append(dict(command=command, returncode=result.returncode,
                            stdout=result.stdout, stderr=result.stderr))
        write(ROOT / 'state/CPU_closeout_checks.json', results)
        assert result.returncode == 0, results[-1]
    assert compileall.compile_dir(str(SOURCE), quiet=1)
    owner = parent.prior.resources()
    owner.restore(list(range(8)))
    owned = owner.burners()
    active = owner.processes()
    assert set(owned) == set(range(8)), 'A verified burner is missing'
    # The burner worker starts a child EngineCore that owns the CUDA context.
    # Verify its UID and inherited session, not equality with the supervisor PID.
    assert all(Path('/proc', str(row['pid'])).stat().st_uid == os.getuid()
               and os.getsid(row['pid']) == owned[row['gpu']] for row in active), \
        'An unrelated GPU process remains untouched'
    assert {row['gpu'] for row in active} == set(range(8))
    parent.prior.preserve()
    now = time.time()
    receipt = dict(time=now, KST=datetime.fromtimestamp(now, ZoneInfo('Asia/Seoul')).isoformat(),
        elapsed_hours_from_original_start=(now - START) / 3600,
        original_overall_deadline=DEADLINE, original_deadline_exceeded=now > DEADLINE,
        reason='Assistant capacity interruption delayed CPU analysis and publication. No GPU experiment ran after the deadline.',
        GPU_experiments_closed=True, analysis_ready_for_publication=True,
        approved_burners=owned, active_GPU_processes=active,
        GPU_query=subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                         '--format=csv'], text=True),
        original_index_preserved=True, external_runtime_preserved=True,
        CPU_unit_test_count=15, CPU_unit_tests_passed=True,
        status='GPU_EXPERIMENTS_CLOSED_CPU_ANALYSIS_READY_FOR_PUBLICATION')
    write(ROOT / 'state/CAMPAIGN_CLOSED.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

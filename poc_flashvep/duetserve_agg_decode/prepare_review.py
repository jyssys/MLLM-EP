"""Prepare completed evidence for agent review; never push automatically."""
import fcntl
import json
import os
import runpy
import subprocess
import time

from .common import PYTHON, QUIET_LOCK, REPO, ROOT, digest, preserve, resources, write


def main():
    write(ROOT / 'state/analysis_controller_active.json', dict(pid=os.getpid(), time=time.time(),
        source_SHA256=digest(__file__), scope='CPU-only preparation controller. It waits for all required GPU receipts.'))
    revised = (ROOT / 'state/user_primary_scope_revision_v2.json').exists()
    required = (('primary_confirmation_complete_v2.json', 'primary_long_windows_complete_v2.json',
                 'primary_secondary_complete_v2.json', 'tp_diagnostic_characterization_complete.json') if revised else
                ('fresh_confirmation_complete.json', 'long_replication_complete.json',
                 'secondary_plan_result.json', 'tp_diagnostic_characterization_complete.json'))
    until = time.monotonic() + 43200
    while not all((ROOT / 'state' / name).exists() for name in required):
        assert time.monotonic() < until, 'Evidence preparation deadline exceeded'
        time.sleep(15)
    if revised:
        assert json.loads((ROOT / 'state/primary_long_windows_complete_v2.json').read_text())['status'] == 'COMPLETE'
        assert json.loads((ROOT / 'state/primary_secondary_complete_v2.json').read_text())['status'] == 'COMPLETE'
    else:
        assert json.loads((ROOT / 'state/long_replication_complete.json').read_text())['passed']
        assert json.loads((ROOT / 'state/secondary_plan_result.json').read_text())['returncode'] == 0
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        owner = resources()
        active, approved = owner.processes(), owner.burners()
        assert all(os.getsid(row['pid']) == approved.get(row['gpu']) for row in active)
        if not active:
            owner.restore(list(range(8)))
        preserve()
        started = time.time()
        try:
            subprocess.run([str(PYTHON), '-m', 'unittest', 'poc_flashvep.duetserve_agg_decode.test_analysis'],
                           cwd=REPO, check=True)
            from .analyze import table
            table()
            runpy.run_module('poc_flashvep.duetserve_agg_decode.work_accounting', run_name='__main__')
            runpy.run_module('poc_flashvep.duetserve_agg_decode.cohort_phases', run_name='__main__')
            from .interference import overhead, mixture_change
            for prefix in (
                'profile_DP2_B8192_S1024_FULL_N128_O200_q2_s2026111616',
                'profile_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111616',
                'profile_DP2_B512_S16_FULL_N128_O200_q6_s2026111616',
                'profile_Native_N128_O200_q6_s2026111616',
            ):
                overhead(prefix)
            from .kernel_trace import analyze as kernel_trace
            for name in (
                'profile_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111616_NSYS_v1',
                'profile_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111616_NSYS_v2',
                'profile_DP2_B512_S16_FULL_N128_O200_q6_s2026111616_NSYS_v1',
                'profile_Native_N128_O200_q6_s2026111616_NSYS_v1',
            ):
                kernel_trace(name)
            mixture_change('profile_DP2_B8192_S1024_FULL_N128_O200_q2_s2026111616_HOST_ONLY_v1',
                           'profile_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111616_HOST_ONLY_v1')
            from .audit import main as audit
            audit()
            from .figures import main as figures
            figures()
            from .confirmed_four_figures import render
            render()
            from .reports import main as reports
            reports()
            preserve()
            write(ROOT / 'state/analysis_ready_for_review.json', dict(start_epoch=started,
                end_epoch=time.time(), status='READY_FOR_AGENT_REVIEW', GPU_work_closed=False,
                push_performed=False,
                scope='Inspect numerical conclusions, figures, metrology labels, and preserved resources before closeout or publication. This process does not mark the research complete.'))
        except BaseException as error:
            write(ROOT / 'state/analysis_preparation_failed.json', dict(time=time.time(), error=repr(error),
                scope='Preserve all generated files. No commit, push, or automatic scientific-success claim occurred.'))
            raise


if __name__ == '__main__':
    main()

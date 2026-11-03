"""Wait for the final controllers, then test and analyze without GPU contention."""
import contextlib
from .common import *
from .analysis_lane import quiet_analysis


def main():
    terminal = {'COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED',
        'NOT_RUN_NO_ACCEPTED_SOURCE_OR_BUDGET'}
    names = ('qualified_load_joint', 'qualified_lower_load_joint', 'matched_aggregated')
    while any(not (p := ROOT/'state'/(n+'.json')).exists()
            or json.loads(p.read_text()).get('stage') not in terminal for n in names):
        assert time.time() < GPU_DEADLINE+60, 'Controller status requires manual review'
        time.sleep(5)
    with quiet_analysis('final_tests_figures_and_archive'):
        preserve()
        write(ROOT/'state/GPU_CAMPAIGN_COMPLETE.json', dict(time=time.time(),
            scope='All scheduled controllers reached terminal status. Publication remains pending.'))
        with (ROOT/'logs/closeout_cpu_tests.log').open('w') as log, contextlib.redirect_stdout(log):
            from . import test_timed_capture, test_bulk_written
            test_timed_capture.main()
            test_bulk_written.ROOT = ROOT/'raw/closeout_cpu_validation'
            test_bulk_written.main()
        from .joint_figures import render
        for name in ('qualified_ouro_s1_nsys128', 'qualified_ouro_s2_nsys128',
                'qualified_ouro_s3_nsys128', 'persistence_ouro_nsys512',
                'window_huginn_s1_nsys384', 'qualified_load_joint_ouro128',
                'qualified_lower_load_joint_ouro128'):
            if (ROOT/'raw'/name/'profile_analysis.json').exists():
                print(render(name), flush=True)
        from .final_analysis import main as analyze
        analyze()
        from .publish_campaign import archive
        archive()
        preserve()
        write(ROOT/'state/closeout_analysis.json', dict(status='COMPLETE', time=time.time()))
        print('CLOSEOUT_ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

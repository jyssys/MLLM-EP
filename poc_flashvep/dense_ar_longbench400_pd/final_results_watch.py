"""Resume analysis only after the original watcher encountered a cache race."""
import time
from .common import ROOT, write, preserve
from .final_confirmation_analysis import main as analyze


def main():
    preserve()
    while True:
        analyze()
        if all((ROOT/'state'/f'final_performance_lane{i}_complete_v1.json').exists() for i in range(4)):
            break
        time.sleep(30)
    write(ROOT/'state/final_transition_complete_v1.json',dict(time=time.time(),complete=True,
        resumed_analysis_only=True,
        previous_log='logs/background_final_transition_wait_and_execute_v1.log',
        scope='The prior analysis coordinator stopped on an intermediate metric-cache schema. '
              'GPU lane controllers continued. No experiment was restarted. '
              'Root scientific review,remaining repairs,figures,tests,resource closeout and publication remain.'))


if __name__=='__main__':
    main()

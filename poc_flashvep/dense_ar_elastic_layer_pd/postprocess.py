"""Restore resources after the frozen grid, then build derived evidence."""
import argparse,importlib
from .common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--monitor-pid',type=int,required=True)
    args=parser.parse_args()
    while Path(f'/proc/{args.monitor_pid}').exists():time.sleep(20)
    terminal=json.loads((ROOT/'state/monitor_terminal.json').read_text())
    assert terminal['status']=='GPU_GRID_FINISHED',terminal
    from .closeout import close
    close()
    stages=('result_analysis','cost_validation','decision_analysis','systems_accounting',
        'secondary_analysis','timeline_analysis','prediction_figures','figures')
    completed=[]
    for stage in stages:
        write(ROOT/'state/postprocess_progress.json',dict(time=time.time(),completed=completed,active=stage))
        importlib.import_module('.'+stage,__package__).main();completed.append(stage)
    write(ROOT/'state/postprocess_progress.json',dict(time=time.time(),completed=completed,status='COMPLETE',
        scope='GPU work is closed. All8 approved burners have fresh completed generation. Reports still require final review and publication.'))
    print('POSTPROCESS_COMPLETE_REPORT_REVIEW_REQUIRED',flush=True)


if __name__=='__main__':main()

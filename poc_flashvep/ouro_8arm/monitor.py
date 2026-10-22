"""Read-only campaign monitoring. It never signals an experiment or uses CUDA."""
import argparse
import datetime
import time
from .contract import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--interval',type=float,default=50);a=p.parse_args()
    while True:
        controller=read(ROOT/'state/continuation_controller.json')
        path=ROOT/'state/continuation_stage.json';stage=read(path) if path.exists() else {'stage':'Static pilot'}
        logs=list((ROOT/'logs').glob('*.log'));latest=max(logs,key=lambda v:v.stat().st_mtime)
        folder=ROOT/'raw'/latest.stem
        count=sum(1 for pattern in ['gsm8k_test_*.json','math500_test_*.json'] for _ in folder.glob(pattern))
        print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S'),
            stage=stage.get('stage'),run=latest.stem,answer_records_saved=count,status=stage.get('status','RUNNING'))),flush=True)
        if stage.get('status')=='NEEDS_REVIEW':return
        error=ROOT/'state/continuation_error.json'
        if error.exists() and error.stat().st_mtime>(ROOT/'state/continuation_controller.json').stat().st_mtime:
            print('CONTROLLER_ERROR',error.read_text(),flush=True);raise SystemExit(2)
        if not Path('/proc',str(controller['pid'])).exists():
            print('CONTROLLER_EXITED_WITHOUT_REVIEW_MARKER',flush=True);raise SystemExit(3)
        time.sleep(a.interval)


if __name__=='__main__':main()

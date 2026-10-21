"""Small actual-answer timing cohort with common no-refill successor scheduling."""
from .contract import *
from .campaign import run
from .lease import restore


def main():
    ids=[f'{domain}/test/{i}' for domain in ['gsm8k','math500'] for i in range(4)]
    plan=read(ROOT/'matched_natural_plan.json');assert plan['ids']==ids and plan['cap']==4096
    status=read(ROOT/'state/dynamic_probe_status.json')
    try:
        for arm in ARMS:
            if arm in status and status[arm]['status']=='BLOCKED':continue
            options=['--ids',','.join(ids)]
            if arm in ['T4','TD']:
                module='tp_runtime'
                if arm=='TD':options+=['--dynamic']
            else:module='request_runtime';options+=['--arm',arm,'--lockstep']
            if arm in ['S4','SD']:restore([5,6,7])
            run('matched_natural_'+arm+'_v1',module,ARMS[arm]['GPUs'],options,distributed=arm in ['T4','TD'],timeout=3600)
    finally:restore([4,5,6,7])


if __name__=='__main__':main()

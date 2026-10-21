"""Run only the 286 additional frozen questions for the user-approved target."""
from .contract import *
from .campaign import run
from .lease import restore


def main():
    manifest=read(ROOT/'evaluation_subset400.json')
    assert len(manifest['extension_ids'])==286
    assert len({r['id'] for r in subset400()})==400
    status=read(ROOT/'state/dynamic_probe_status.json')
    try:
        for arm in ['S4','L4','O4','T4','SD','LD','OD','TD']:
            if arm in status and status[arm]['status']=='BLOCKED':continue
            args=['--extension400']
            if arm in ['T4','TD']:
                if arm=='TD':args+=['--dynamic']
                run('extension400_'+arm+'_v1','tp_runtime',args=args,distributed=True,timeout=28800)
            else:
                run('extension400_'+arm+'_v1','request_runtime',ARMS[arm]['GPUs'],['--arm',arm]+args,timeout=28800)
        for tp in [1,4]:
            run(f'native_TP{tp}_extension400_v1','native_quality',[4] if tp==1 else [4,5,6,7],
                ['--tp',str(tp),'--extension400'],native=True,timeout=10800)
    finally:restore([4,5,6,7])


if __name__=='__main__':main()

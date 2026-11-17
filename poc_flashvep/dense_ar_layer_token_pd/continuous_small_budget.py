"""Check smaller legal mixed quanta instead of assuming serialization wins."""
import fcntl
from .common import ROOT, QUIET_LOCK, resources, save, write
from .continuous_frontier import configuration
from .run import execute
from .analysis import metrics


def main():
    values=[]
    try:
        for budget in (64,128):
            cfg=configuration(f'small_mixed_Token_B{budget}_S2048_D8_N64_O16_q4.8_v1',
                budget=budget,D_admission_batch=8,D_slots=8,
                explicit_cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256,512])
            if not execute(save(cfg)):
                continue
            result=metrics(cfg['name'])
            assert result['output_gate']['passed']
            values.append(result)
            write(ROOT/'analysis/small_continuous_token_budget_frontier.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()

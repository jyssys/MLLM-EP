"""Measure serial request costs without state hashing or CPU staging."""
import fcntl,json
from .common import *
from .prepare import save
from .run import execute
from .analyze import metrics


def main():
    jobs=[]
    for label,mode,cut in [('native','STATE_REFERENCE',32),('span','SPAN_ZERO',32),
                            ('c35','MIGRATE',35),('c34','MIGRATE',34),('c32','MIGRATE',32),('c28','MIGRATE',28)]:
        c=json.loads((ROOT/'configs/layer_qwen8_L8000_O16_state_reference_c36_N8_gpu_v6.json').read_text())
        c.update(name=f'cost_qwen8_L8000_O16_N8_{label}_events_v1',layer_mode=mode,layer_cut=cut,
            state_capture=False,hidden_verify_hash=False,save_state_tensors=False,
            partial_KV_wire=mode=='MIGRATE',execution_events=True,batch_commands=True,
            minimal_serving_trace=False,clean_timing=False,phase='SERIAL_CUDA_EVENT_COST_CALIBRATION',
            scope='Independent serial request calibration. Stream event enclosures are not kernel unions. No model state hashing. Not clean online performance.')
        jobs.append(save(c))
    write(ROOT/'state/serial_layer_cost_plan.json',dict(jobs=jobs))
    try:
        for n in jobs:
            if not execute(n,restore=False):break
            m=metrics(n);print('COST',n,m['TTFT_p50_ms'],flush=True)
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()

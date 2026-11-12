"""Measure source and destination enclosures at the online batch geometry."""
import fcntl,json
from .common import *
from .prepare import save
from .run import execute


def main():
    plan=json.loads((ROOT/'state/serial_layer_cost_plan.json').read_text());jobs=[]
    for old in plan['jobs']:
        c=json.loads((ROOT/'configs'/(old+'.json')).read_text())
        c.update(name=old.replace('N8','N16').replace('events_v1','batch4_events_v1'),count=16,
            request_caps=[16]*16,arrival_offsets=[0.]*16,calibration_serial=False,
            phase='MATCHED_BATCH4_CUDA_EVENT_CALIBRATION',
            scope='Four source slots, sixteen simultaneous requests. Actual per-forward batch size is checked in exported events. Not clean online serving or kernel union time.')
        jobs.append(save(c))
    write(ROOT/'state/batch4_layer_cost_plan.json',dict(jobs=jobs))
    try:
        for n in jobs:
            if not execute(n,restore=False):break
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()

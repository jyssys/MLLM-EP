"""Separate generic import serialization from Layer migration benefit."""
import json
from .common import *
from .prepare import save
from .run import execute
from .analyze import metrics,compare
from .resume_campaign import restore,record


def main():
    template=json.loads((ROOT/'configs/eager_serving_qwen8_L8000_O16_N64_s3_guard_v22.json').read_text())
    groups=json.loads((ROOT/'state/eager_confirmation_plan_v23.json').read_text())['groups']
    jobs=[]
    for group in groups:
        native=json.loads((ROOT/'configs'/(group['arms']['native']+'.json')).read_text())
        c=dict(template,name=f'eager_guard_qwen8_L8000_O16_N128_s{group["seed"]%100}_ablation_v26',
            count=128,request_caps=[16]*128,arrival_seed=group['seed'],arrival_offsets=native['arrival_offsets'],
            timeout=360,phase='POST_CONFIRMATION_GENERIC_ADMISSION_ABLATION',
            scope='Same arrival seed and input as the main confirmation. This ablation runs later, not inside the original balanced order. It isolates the configured generic guard, but time-order confounding remains possible.')
        jobs.append((save(c),group['arms']['native']))
    write(ROOT/'state/native_guard_ablation_plan_v26.json',dict(jobs=[n for n,_ in jobs]))
    try:
        for name,ref in jobs:
            if not execute(name,restore=False):return
            gate=compare(ref,name)
            record('native_guard_ablation',dict(name=name,metrics=metrics(name),output_gate=gate))
            if not gate['passed']:return
    finally:restore()


if __name__=='__main__':main()

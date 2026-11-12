"""Qualify a full-depth no-migration control for the new static span forward."""
import json
from .common import *
from .prepare import save


def main():
    jobs=[]
    for count in (8,32):
        source=ROOT/f'configs/layer_qwen8_L8000_O16_state_reference_c36_N{count}_gpu_v6.json'
        c=json.loads(source.read_text());c.update(name=f'layer_qwen8_L8000_O16_span_zero_N{count}_qual_v7',
            layer_mode='SPAN_ZERO',partial_KV_wire=False,hidden_verify_hash=False,batch_commands=True,
            phase='FULL_DEPTH_STATIC_SPAN_NO_MIGRATION_QUALIFICATION')
        jobs.append(save(c))
    write(ROOT/'state/span_zero_qualification_plan.json',dict(jobs=jobs,
        scope='Full 36-layer source execution, native full KV handoff, and native full Decode. No Prefill layers migrate. This control separates forward implementation and cohort scheduling from Layer offloading.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()

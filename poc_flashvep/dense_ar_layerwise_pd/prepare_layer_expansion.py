"""Expand only after actual N8 state and generated-token qualification."""
import json
from .common import *
from .prepare import base,save


def main():
    initialize();assert json.loads((ROOT/'analysis/layer_N8_numerical_qualification.json').read_text())['passed']
    jobs=[]
    # Same native source Prefix geometry. D serialization changes residency,
    # not the simultaneous N32 source submission contract.
    c=base('layer_qwen8_L1024_O16_state_reference_N32_v4',count=32,length=1024,cap=16)
    c.update(dataset='qualification_qwen8_L1024.json',layer_mode='STATE_REFERENCE',layer_cut=32,
        state_capture=True,serialize_D_imports=True,token_budget=4096,phase='N32_STATE_REFERENCE',
        timeout=300)
    jobs.append(save(c))
    for count in (8,32):
        for cut in (35,34,32,28):
            c=base(f'layer_qwen8_L1024_O16_migrate_c{cut}_N{count}_v4',count=count,length=1024,cap=16)
            c.update(dataset='qualification_qwen8_L1024.json',layer_mode='MIGRATE',layer_cut=cut,
                state_capture=True,partial_KV_wire=True,serialize_D_imports=True,token_budget=4096,
                calibration_serial=count==8,phase='LAYER_CUT_EXPANSION_CORRECTNESS',timeout=300,
                scope='Simultaneous N32 submission does not mean 32 GPU-resident requests. Destination is serialized for State qualification. This is not serving performance.')
            jobs.append(save(c))
    write(ROOT/'state/layer_expansion_plan.json',dict(jobs=jobs,
        scope='All four tail Layer groups receive N8 before N32. No online performance claim. Outputs must equal independent Native; valid queried states are checked separately.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()

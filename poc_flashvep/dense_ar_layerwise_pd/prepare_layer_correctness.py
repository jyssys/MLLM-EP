"""Separate local layer-state checks from actual 2GPU continuation."""
import json
from .common import *
from .prepare import base,save


def main():
    initialize();assert json.loads((ROOT/'analysis/native_N32_qualification.json').read_text())['passed']
    rows=json.loads((ROOT/'data/qualification_qwen8_L8000.json').read_text())
    for row in rows:row['input_ids']['qwen8']=row['input_ids']['qwen8'][:1024]
    path=ROOT/'data/qualification_qwen8_L1024.json'
    if path.exists():assert json.loads(path.read_text())==rows
    else:write(path,rows)
    jobs=[]
    for mode in ('STATE_REFERENCE','LOCAL_SPLIT','MIGRATE'):
        c=base('layer_qwen8_L1024_O16_'+mode.lower()+'_c32_N8_v1',length=1024,cap=16)
        c.update(dataset=path.name,layer_mode=mode,layer_cut=32,state_capture=True,
            partial_KV_wire=mode=='MIGRATE',serialize_D_imports=True,
            phase='LAYER_STATE_CORRECTNESS',calibration_serial=True,timeout=300,
            transport_scope='CPU-staged Hidden and Residual. Native NIXL prefix-KV transfer. Not a fast transport performance result.')
        jobs.append(save(c))
    write(ROOT/'state/layer_initial_plan.json',dict(jobs=jobs,
        scope='N8 serial diagnostic: exact all-prompt queried KV and Hidden. Local split precedes actual partial P-to-D execution.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()

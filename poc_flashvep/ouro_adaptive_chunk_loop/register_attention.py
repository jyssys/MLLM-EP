"""Target the lost earlier chunk without changing benchmark selection."""
import json
from .common import ROOT,write

def main():
    original=json.loads((ROOT/'p1_controlled_plan.json').read_text())
    jobs=[]
    for gpu,old in enumerate(j for j in original['jobs'] if j['kind']=='diagnostic'):
        jobs.append(dict(old,id=f'p1_attention_boundary_graph{int(old["graphs"])}_v1',gpu=gpu,
            capture_attention=True,layers=[0,1,5,11,17,23]))
    value=dict(gpus=[0,1],jobs=jobs,
        selection_reason='The earliest changed prefix KV appears at depth2/layer1. Earlier chunk rows were overwritten in the first observer. Capture each chunk by absolute positions.',
        scope='Common-prefix operator diagnostic. No quality or performance claim.',
        prior_plan_preserved='p1_controlled_plan.json')
    path=ROOT/'p1_attention_probe_plan.json'
    assert not path.exists()
    write(path,value)

if __name__=='__main__':
    main()

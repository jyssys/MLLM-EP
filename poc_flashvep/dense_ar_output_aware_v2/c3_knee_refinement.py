"""Add the Native-only capacity-boundary load without deleting old points."""
import json
import time
from .common import ROOT,write,digest
from .c3_campaign import qualified_registry,performance,launch,wait_file

def run():
    wait_file(ROOT/'state/c3_pilot_and_geometry_complete_v2.json')
    amendment=json.loads((ROOT/'configs/c3_confirmation_knee_refinement_v3.json').read_text())
    old=json.loads((ROOT/'jobs/c3_clean_geometry_discovery_v2.json').read_text());registry=qualified_registry()
    labels=list(dict.fromkeys(b['label'] for b in old['blocks']));qps=amendment['added_geometry_and_policy_discovery_QPS']
    lanes=[[],[],[],[]];blocks=[];pair=[0,1]
    for label in labels:
        name=f'V2_C3Geometry_q{qps}_{label}_Discovery208_KneeRefinementV3'
        performance(registry[label],name,'balanced400_v2_discovery.json',pair,qps,2026112262)
        lanes[0].append(name);blocks.append(dict(qps=qps,label=label,name=name,pair=pair,gate=registry[label]['gate']))
    filename='c3_geometry_knee_refinement_v3.json'
    write(ROOT/'jobs'/filename,dict(time=time.time(),lanes=lanes,blocks=blocks,stop_on_failure=False,
        amendment_SHA256=digest(ROOT/'configs/c3_confirmation_knee_refinement_v3.json'),
        scope='Additional Native-capacity-boundary discovery. Keep the same candidate list,discovery208,seed and within-load GPU pair. '
            'The original2.6/4.8 runs remain in tuning and publication. No heldout request enters tuning.'))
    launch(filename);write(ROOT/'state/c3_geometry_knee_refinement_complete_v3.json',dict(time=time.time(),complete=True,plan=filename))

if __name__=='__main__':run()

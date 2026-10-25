"""Validate a second random-control design and the final strict Foil loader."""
import json
import subprocess
import time
from .common import ROOT,write
from .final_checks import dispatch

def main():
    while True:
        records=[json.loads(s) for s in (ROOT/'progress.jsonl').read_text().splitlines()]
        if any(r['event']=='FINAL_MEASUREMENT_CONTROLS_COMPLETE' for r in records):
            break
        time.sleep(5)
    rows=json.loads((ROOT/'workload_manifest.json').read_text())['rows']
    selected=[r['id'] for r in rows if r['dataset']=='language' and r['split'] in ('calibration','heldout')]
    arms=[f'O1_drop_{q}_renorm0' for q in [.05,.1,.2,.3]]+[f'N1_independent_site_drop_{q}' for q in [.05,.1,.2,.3]]
    jobs=[dict(id=f'{model}_independent_control_shard{shard}_v1',model=model,phase='p2',ids=selected[shard::8],
        arms=arms,length=512,timeout_s=21600) for model in ['ifm_ouro','ifm_huginn','foil1'] for shard in range(8)]
    path=ROOT/'plan_independent_controls.json'
    write(path,dict(phase='independent_control',jobs=jobs))
    dispatch(path,'0,1,2,3,4,5,6,7')
    path=ROOT/'plan_strict_foil_load.json'
    write(path,dict(phase='p0',jobs=[dict(id='foil1_strict_load_final_v1',model='foil1',phase='p0')]))
    dispatch(path,'0')

if __name__=='__main__':
    main()

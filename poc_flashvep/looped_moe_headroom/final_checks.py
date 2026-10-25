"""Run measurement controls after the natural-generation campaign."""
import json
import subprocess
import time
from .common import ROOT,REPO,SEED,write,record

def dispatch(path,gpus):
    with (ROOT/'logs'/f'dispatch_{path.stem}.log').open('a') as f:
        subprocess.run(['/home/esjung/anaconda3/bin/python','-m','poc_flashvep.looped_moe_headroom.dispatch',str(path),'--gpus',gpus],
            cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    prerequisite=json.loads((ROOT/'plan_generation_quality.json').read_text())
    while not all((ROOT/'raw'/j['id']/'complete.json').exists() for j in prerequisite['jobs']):
        failed=[j['id'] for j in prerequisite['jobs'] if (ROOT/'raw'/j['id']/'failure.json').exists()]
        assert not failed,failed
        time.sleep(5)
    rows=json.loads((ROOT/'workload_manifest.json').read_text())['rows']
    ids=[r['id'] for r in rows if r['dataset']=='language' and r['split']=='smoke'][:1]
    micro=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        for repeat in range(3):
            micro.append(dict(id=f'{model}_micro_v2_start{repeat}',model=model,phase='micro',ids=ids,order_seed=SEED+repeat))
    path=ROOT/'plan_micro_v2.json'
    write(path,dict(phase='micro',jobs=micro))
    dispatch(path,'0')
    profile=[dict(id=f'{model}_heavy_profile_v1',model=model,phase='profile',ids=ids) for model in ['ifm_ouro','ifm_huginn','foil1']]
    path=ROOT/'plan_heavy_profile.json'
    write(path,dict(phase='profile',jobs=profile))
    dispatch(path,'0')
    for participants in [1,8]:
        for repeat in range(3):
            jobs=[dict(id=f'ifm_ouro_contention_p{participants}_r{repeat}_gpu{gpu}_v1',model='ifm_ouro',phase='contention',
                ids=ids,gpus=[gpu],barrier=f'barrier_p{participants}_r{repeat}_v1',participants=participants)
                for gpu in range(participants)]
            path=ROOT/f'plan_contention_p{participants}_r{repeat}.json'
            write(path,dict(phase='contention',jobs=jobs))
            dispatch(path,','.join(map(str,range(participants))))
    record('FINAL_MEASUREMENT_CONTROLS_COMPLETE')

if __name__=='__main__':
    main()

"""Freeze quiet corrected confirmation before reading FINAL_ONLINE outcomes."""
import copy
import json
import random
from .common import ROOT,write

def main():
    target=ROOT/'p5_quiet_confirmation_plan.json';assert not target.exists()
    training=json.loads((ROOT/'p3_budget_pilot_plan.json').read_text())
    groups=[]
    for start in range(3):
        for shape,(P,N) in enumerate(((2033,4),(8177,8))):
            for index,B in enumerate((40,120)):
                template=next(j for j in training['jobs'] if j['shape']==[P,N] and
                    any(c['label']=='J1' and c['budget_ms']==B for c in j['cells']))
                gpu=(start*3+shape*2+index)%8
                job=copy.deepcopy(template)
                job.update(id=f'cb_p5_confirm_P{P}_N{N}_B{B}_S{start}_v2',gpu=gpu,split='FINAL_ONLINE',
                    trials=[2*start,2*start+1],warm_split='CALIBRATION',warm_trial=0,cap=512,
                    observe=False,quiet=True,allow_any_gpu=False,timeout_s=3600)
                random.Random(20261023+2700+start*100+shape*10+index).shuffle(job['cells'])
                groups.append(dict(jobs=[job],barrier=None))
    write(target,dict(gpus=list(range(8)),groups=groups,
        scope='Quiet one-GPU-at-a-time independent starts. Same policy parameters. Fresh FINAL_ONLINE families. Natural EOS allowed, cap512.',
        selection='TRAIN budget40 feasibility contrast and budget120 amortized contrast. Both shapes retained. No final outcome used.',
        qualification='Focused confirmation, not a sustainable-capacity experiment. Other GPUs remain isolation-reserved.'))

if __name__=='__main__':
    main()

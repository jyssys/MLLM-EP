"""Matched-device 1/2/4/8-worker calibration with explicit warm barriers."""
from .common import ROOT,write

def main():
    groups=[]
    for count in (1,2,4,8):
        for base in range(0,8,count):
            barrier=f'cal_v1_w{count}_b{base}'
            jobs=[dict(id=f'{barrier}_g{gpu}',gpu=gpu,kind='timing',
                shapes=[[256,4],[2048,8]],groups=[1],graphs=True,**{'async':True},
                cap=48,repeats=3,barrier=barrier,lane=f'CALIBRATION_W{count}',timeout_s=3600)
                for gpu in range(base,base+count)]
            groups.append(dict(barrier=barrier,jobs=jobs))
    write(ROOT/'contention_plan.json',dict(gpus=list(range(8)),groups=groups))

if __name__=='__main__':main()

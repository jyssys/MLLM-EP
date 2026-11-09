"""A bounded clean protocol pilot precedes held-out confirmation."""
from .policy import *


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        for arm in ('native','zero','loop','tile'):
            cfg=configured(model,f'quick_{model}_{arm}_clean32',arm,32)
            cfg['phase']='N32_CLEAN_POLICY_PILOT_NOT_HELD_OUT_CONFIRMATION'
            save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];i=queue.index(active)+1
    write(p,queue[:i]+jobs+queue[i:])


if __name__=='__main__':main()

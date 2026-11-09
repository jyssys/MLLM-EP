"""One-factor stream controls after the exact stream-ownership tests."""
from .early_online import *


def main():
    prepare();names=[]
    for model in ('ouro','huginn'):
        for arm in ('native','zero','copy_off','copy_on'):
            cfg=config_for(model,'native' if arm=='native' else 'dynamic_loop')
            cfg['name']=f'rc_{model}_{arm}_burst8'
            cfg['phase']='N8_STREAM_AND_FAST_FALLBACK_CONTROL_NOT_FINAL_PERFORMANCE'
            if arm=='zero':cfg['intervention_method']='zero'
            cfg['dedicated_handoff_copy_stream']=arm in ('zero','copy_on')
            cfg['copy_stream_hypothesis']='An independent event-ordered hidden copy removes unrelated P-core work from the snapshot copy stream.'
            save(cfg);names.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];i=queue.index(active)+1
    write(p,queue[:i]+names+queue[i:])


if __name__=='__main__':main()

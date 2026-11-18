"""Confirm every fixed pair/output at one prespecified common load."""
import json
import time
from .common import ROOT,write
from .campaign import job,run_job

SEEDS=(2026111790,2026111791,2026111792)


def freeze():
    path=ROOT/'state/confirmation_design_before_performance.json'
    value=dict(QPS=6.,outputs=[16,32,64,128,200],layers=[4,8,12],
        count=144,admission=8,seeds=list(SEEDS),starts_per_point=3,
        methods=['Agg','Native','Layer4','Token4','Layer8','Token8','Layer12','Token12'],
        prompt_pool='matched144.json',
        scope='Confirm all fixed pairs and outputs at QPS6, without selecting a favorable cut or output. Fresh paired arrival seeds. The same declared real-document pool is used. This is arrival/start confirmation, not an unseen prompt-pool claim. Discovery is not pooled into confirmation confidence intervals.')
    if path.exists():
        assert {k:v for k,v in json.loads(path.read_text()).items() if k!='frozen_epoch'}==value
    else:
        write(path,dict(frozen_epoch=time.time(),**value))


def main():
    freeze()
    while not (ROOT/'state/discovery_complete.json').exists():
        time.sleep(15)
    configs=[]
    methods=[('Agg',0),('Native',0)]+[(axis,m) for m in (4,8,12) for axis in ('Layer','Token')]
    for rep,seed in enumerate(SEEDS):
        for index,cap in enumerate((200,16,64,32,128)):
            shift=(rep+index)%len(methods)
            order=methods[shift:]+methods[:shift]
            for method,layers in order:
                configs.append(job(method,layers,144,cap,6.,seed,stage='confirmation'))
    write(ROOT/'state/confirmation_manifest.json',dict(expected=[c['name'] for c in configs],
        total=len(configs),scope='Three independent starts at each prespecified point.'))
    for index,cfg in enumerate(configs):
        run_job(cfg)
        write(ROOT/'state/confirmation_progress.json',dict(time=time.time(),
            completed=index+1,total=len(configs),last=cfg['name']))
    write(ROOT/'state/confirmation_complete.json',dict(time=time.time(),total=len(configs)))


if __name__=='__main__':
    main()

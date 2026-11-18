"""Confirm the prespecified Last12 cohort-geometry control, not a new policy."""
import json
import time
from .common import ROOT, write
from .campaign import job, run_job
from .host_paths import analyze


def main():
    design=dict(count=36,QPS=6.,outputs=[16,200],layers=12,admission_caps=[1,8],
        seeds=[2026111787,2026111788,2026111789],methods=['Native','Layer','Token'],
        original_axis_seed=2026111787,new_starts=28,
        reason='The primary runs show different realized D cohorts under the same cap8. Last12 is the user-specified default motivation pair. Confirm this fixed pair before inspecting its original admission-control outcomes.',
        scope='Keep the primary cap8 matrix unchanged. The original seed already includes Layer/Token. Add Native at that seed and all three methods at two new seeds. Use N36 exact matched rows and the same declared prompt subset. These are controlled admission experiments, not additional primary QPS points or unseen prompts.')
    path=ROOT/'state/admission_confirmation_design.json'
    if path.exists():
        assert {k:v for k,v in json.loads(path.read_text()).items() if k!='frozen_epoch'}==design
    else:
        write(path,dict(frozen_epoch=time.time(),**design))
    while not (ROOT/'state/diagnostics_complete.json').exists():
        time.sleep(15)
    configs=[]
    for seed in design['seeds']:
        methods=('Native',) if seed==2026111787 else ('Native','Layer','Token')
        for cap in design['outputs']:
            for limit in design['admission_caps']:
                for method in methods:
                    configs.append(job(method,12 if method!='Native' else 0,36,cap,6.,seed,
                        stage='admission_control',dataset='matched144.json',admission=limit))
    assert len(configs)==28
    write(ROOT/'state/admission_confirmation_manifest.json',dict(expected=[c['name'] for c in configs],total=len(configs)))
    for index,cfg in enumerate(configs):
        run_job(cfg);analyze(cfg['name'])
        write(ROOT/'state/admission_confirmation_progress.json',dict(time=time.time(),completed=index+1,total=len(configs),last=cfg['name']))
    write(ROOT/'state/admission_confirmation_complete.json',dict(time=time.time(),total=len(configs)))


if __name__=='__main__':
    main()

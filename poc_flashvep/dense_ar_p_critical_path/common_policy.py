"""Separate a common logical P/D policy from the strong best-tuned arms."""
import json
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write,digest,bind

PLAN='coordinator_common_policy_qualification_r2.json'


def prepare():
    runner=bind();items=[]
    cases=(('NativeCommon16','NativeCommon2','NativeCommon'),('StockCommon16','StockToken','StockToken'),
        ('HybridCommon16','Hybrid','Hybrid'),('HybridIsoCommon16','HybridIso8Approx','Hybrid'))
    for label,parent,arm in cases:
        jobs=[]
        for count,data in ((32,'qualification32.json'),(400,'longbench_balanced400_v2.json')):
            cfg=json.loads((ROOT/'configs'/f'PC_CoordQual_{parent}_Natural400.json').read_text())
            name=f'PC_CommonPolicy_{label}_Natural{count}'
            cfg.update(name=name,dataset=data,dataset_SHA256=digest(ROOT/'data'/data),count=count,
                continuous_source_credit_limit=16,D_running_admission_cap=2,
                defer_D_remote_setup=True,defer_D_request_registration=True,
                seed=2026112371,arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(2026112371).exponential(1/3.2,count-1))].tolist())
            if arm=='StockToken':
                from ..dense_ar_output_aware_v2.work_contract import regenerate
                regenerate(cfg,json.loads((ROOT/'data'/data).read_text()))
            runner.save(cfg);jobs.append(name)
        items.append(dict(label=label,arm=arm,lane=1,pair=[2,3],teacher=None,strict=None,jobs=jobs,
            gate='pc_coord_'+label.lower()+'_composition_r2.json',signature=runner.signature(cfg),
            source_capsule_SHA256=cfg['source_capsule_SHA256'],
            state_scope='Fresh Natural32/400 policy state,lifetime,coverage and task-quality gates. '
                'The model/partition numerical path and its strict-state proof remain inherited from the same coordinator source family.'))
    path=ROOT/'configs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),items=items,state_prefix='pc_common_source_policy',source_capsule_SHA256=None,
        observation='Original Layer uses source window16. Original Hybrid uses8. Strong Stock uses0 and D cap0. '
            'A common D-only Native reference does not remove the source-window confound.',
        scope='Common logical source window16,D cap2,remote-setup deferral and registration deferral. '
            'Keep all strong best-tuned arms. Do not replace Stock cap0 with this mechanism control. '
            'Same policy does not imply equal actual batch,KV capacity use or hidden-buffer ownership. '
            'This tests policy sensitivity and iso-GEMM geometry at the same explicit limits. '
            'No new active-decode insertion or early credit release.'))
    log=ROOT/'logs/coordinator_common_policy_qualification_r2.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.common_policy','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/coordinator_common_policy_controller.json',dict(time=time.time(),pid=child.pid))
    return child.pid


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':print(prepare(),flush=True)
    else:
        bind()
        from ..dense_ar_output_aware_v2.c3_family import run
        run(1,PLAN)

"""Use one physical pair and the published request traces for P attribution."""
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,bind,dispatch_allowed
from .bootstrap import baseline
from .capsule import build


def prepare():
    runner=bind();tests=[];jobs=[]
    for start,seed in enumerate((2026112281,2026112282,2026112283)):
        order=(('Layer','StockToken','Hybrid'),('Hybrid','StockToken','Layer'),('StockToken','Layer','Hybrid'))[start]
        for label in order:
            names=[]
            for stage in ('Clean1','Profile','Clean2'):
                name=f'PC_SamePairP_{label}_s{start}_{stage}'
                cfg=baseline(label,name,[0,1],seed=seed,save_cfg=False)
                if label!='StockToken':
                    source=build(cfg['source_capsule'].removeprefix('pc_'),'r2')
                    cfg.update(source_capsule=source.name,source_capsule_SHA256=digest(source),
                        qualification_gate_v2=f'pc_{label.lower()}_live_observer_composition_r2.json',
                        P_event_timing=stage=='Profile',P_critical_trace=False,source_credit_release_on_retirement=False)
                cfg.update(kernel_activity=stage=='Profile',kernel_activity_roles=['prefill'],
                    collector_namespace='kernel_only_collector_v3',runtime_api_activity=False,activity_copies=False,
                    operator_correlation=False,graph_enclosures=False,activity_delay_seconds=0,activity_ROI_seconds=None,
                    kernel_activity_seconds=86400,observer_scope='Same physical0/1 pair for all axes. Published three arrival seeds. '
                        'Full-trace P kernel-only capture. Optional original-model CUDA events only for separately qualified Layer/Hybrid. '
                        'Retirement tracking and early release are disabled. Stock retains its original compiled method.')
                runner.save(cfg);names.append(name);jobs.append(name)
            tests.append(dict(label=label,start=start,seed=seed,pair=[0,1],jobs=names,
                family='PUBLISHED_TRACE_REPRODUCTION_SAME_PAIR'))
    path=ROOT/'jobs/same_pair_p_critical_cpc_r4.json';assert not path.exists()
    write(path,dict(time=time.time(),lanes=[jobs,[],[],[]],tests=tests,stop_on_failure=True,
        independent_triplets_per_arm=3,observer_fraction=.05,exact_output_work=True,
        scope='Prospective replacement of unlaunched multi-pair P event profiles. Keep all older plans and stop records. '
            'All axes use identical prompts,output work,arrival traces and physical pair. No request/token bootstrap. '
            'The separate original-Stock6/7 CPC is calibration,not additional starts in this paired comparison. '
            'Mean P GPU resource demand is not the additive decomposition of a historical host mean. '
            'Stock has no per-request forward event attribution after its strict observer failure. Keep that limitation explicit.'))
    log=ROOT/'logs/same_pair_p_critical_cpc_r4.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.same_pair_profile','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/same_pair_profile_controller.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def run():
    runner=bind()
    for label in ('layer','hybrid'):
        path=ROOT/'state'/f'pc_{label}_live_observer_composition_r2.json'
        while not path.exists():
            assert dispatch_allowed();time.sleep(10)
        assert json.loads(path.read_text())['passed'],path
    runner.lane('same_pair_p_critical_cpc_r4.json',0)


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(),flush=True)

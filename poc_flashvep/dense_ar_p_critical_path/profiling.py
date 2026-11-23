"""Freeze full-trace kernel/event CPC tests before their results exist."""
import argparse
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,bind,dispatch_allowed
from .bootstrap import baseline
from .capsule import build,collector


def prepare():
    runner=bind();collector();tests=[];lanes=[[],[],[],[]]
    for label,lane in (('Layer',0),('Hybrid',1)):
        gate=f'pc_{label.lower()}_live_observer_composition_r2.json'
        original=json.loads((ROOT/'configs'/f'PC_Metadata_{label}.json').read_text())
        parent=json.loads((ROOT/'references/source_capsules'/original['source_capsule'].replace('.zip','.json')).read_text())
        source=build(parent['source'].split('/')[-1],'r2');pair=[2*lane,2*lane+1]
        for start,seed in enumerate((2026112281,2026112309,2026112310)):
            names=[]
            for stage in ('Clean1','Profile','Clean2'):
                name=f'PC_P_CPC3_{label}_s{start}_{stage}'
                cfg=baseline(label,name,pair,seed=seed,save_cfg=False)
                cfg.update(source_capsule=source.name,source_capsule_SHA256=digest(source),qualification_gate_v2=gate,
                    P_critical_trace=True,P_event_timing=stage=='Profile',kernel_activity=stage=='Profile',
                    kernel_activity_roles=['prefill'],collector_namespace='kernel_only_collector_v3',
                    runtime_api_activity=False,activity_copies=False,operator_correlation=False,graph_enclosures=False,
                    activity_delay_seconds=0,activity_ROI_seconds=None,kernel_activity_seconds=86400,
                    observer_scope='Full measured trace. P-only kernel collection and original-model CUDA-event enclosures. '
                        'No mid-trace synchronization. Require exact output work and the frozen5% CPC gate.')
                if label!='StockToken':cfg['source_credit_release_on_retirement']=False
                natural=json.loads((ROOT/'configs'/f'PC_LiveNamespace_{label}_Natural400_R2.json').read_text())
                assert runner.signature(cfg)==runner.signature(natural),(label,runner.signature(cfg),runner.signature(natural))
                runner.save(cfg);lanes[lane].append(name);names.append(name)
            tests.append(dict(label=label,start=start,seed=seed,pair=pair,jobs=names,qualification_gate=gate))
    target=ROOT/'jobs/p_only_fulltrace_cpc_r3.json';assert not target.exists()
    write(target,dict(time=time.time(),lanes=lanes,tests=tests,stop_on_failure=True,
        observer_fraction=.05,exact_output_work=True,independent_triplets_per_arm=3,
        scope='All three CPC triplets are declared before observation. Failed gates remain diagnostic. '
            'Do not rerun until a favorable observer ratio appears. Event enclosure is not exclusive kernel time. '
            'D operator time,DMA,SM utilization and HBM bandwidth remain NOT_MEASURED in this P-only capture.'))
    children=[]
    for label,lane in (('Layer',0),('Hybrid',1)):
        log=ROOT/'logs'/f'p_only_cpc_r3_lane{lane}.log';assert not log.exists()
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.profiling',
                'await_run','--lane',str(lane)],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(label=label,lane=lane,pid=child.pid))
    write(ROOT/'state/p_only_cpc_r3_controller.json',dict(time=time.time(),children=children))
    return children


def await_run(lane):
    runner=bind();plan=json.loads((ROOT/'jobs/p_only_fulltrace_cpc_r3.json').read_text())
    gate=next(v['qualification_gate'] for v in plan['tests'] if v['pair']==[2*lane,2*lane+1])
    path=ROOT/'state'/gate
    while not path.exists():
        assert dispatch_allowed(),'No new capture after dispatch cutoff'
        # The new source has a separate mandatory full400 observer proof.
        time.sleep(10)
    assert json.loads(path.read_text())['passed'],path
    runner.lane('p_only_fulltrace_cpc_r3.json',lane)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','await_run'));p.add_argument('--lane',type=int)
    args=p.parse_args();print(prepare() if args.action=='prepare' else await_run(args.lane),flush=True)

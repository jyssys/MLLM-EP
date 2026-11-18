"""Resume a fixed-partition experiment matrix without silent retries."""
import argparse
import fcntl
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,QUIET_LOCK,initialize,resources,save,write
from .configuration import base
from .prepare import pools,historical_outputs
from .run import execute
from .analysis import metrics
from .recovery import resolve_configuration


def key(method,layers,count,cap,qps,seed,stage='discovery',admission=8):
    version=3 if (ROOT/'state/design_freeze_v3_before_performance.json').exists() else 1
    return f'{stage}_{method}_Last{layers if method in ("Layer","Token") else 0}_N{count}_O{cap}_q{qps:g}_s{seed}_A{admission}_v{version}'


def job(method,layers,count,cap,qps,seed,stage='discovery',dataset=None,admission=8,**changes):
    name=key(method,layers,count,cap,qps,seed,stage,admission)
    dataset=dataset or ('matched144.json' if count==144 else 'discovery72_declared_cycle.json')
    cfg=base(name,method,layers,count,cap,qps,seed,dataset,
        phase='UNIFIED_MATCHED_AXIS_'+stage.upper(),D_admission_batch=admission,
        reference_name='HISTORICAL_original_outputs',**changes)
    cfg['matrix_stage']=stage
    if (ROOT/'state/design_freeze_v3_before_performance.json').exists():
        cfg.update(common_prospective_KV_reservation=method!='Agg',execution_contract_version=3)
    existing=ROOT/'configs'/(name+'.json')
    if existing.exists():
        assert json.loads(existing.read_text())==cfg
    return cfg


def run_job(cfg):
    cfg=resolve_configuration(cfg)
    receipt=ROOT/'raw'/cfg['name']/'dispatcher_result.json'
    cached=ROOT/'analysis'/(cfg['name']+'_metrics.json')
    if receipt.exists() and cached.exists() and json.loads(receipt.read_text()).get('complete'):
        value=json.loads(cached.read_text())
        if value.get('analysis_schema_version')==3 and value['output_gate']['passed'] and value['unified_worker_gate']:
            return value
    write(ROOT/'state/active_job.json',dict(time=time.time(),name=cfg['name'],
        stage=cfg['matrix_stage'],method=cfg['evaluation_arm'],layers=cfg['static_layers'],
        cap=cfg['cap'],QPS=cfg['arrival_rate'],count=cfg['count']))
    assert execute(save(cfg)), 'Preserve failed run; repair and use a new version: '+cfg['name']
    value=metrics(cfg['name'])
    assert value['output_gate']['passed']
    return value


def matrix():
    """Output200 gets the complete QPS grid. Other caps start coarse."""
    coarse=[]
    # Compare Output16 and200 early, before filling the intermediate outputs.
    for qps in (2.,6.,10.,4.,8.):
        for cap in (200,16,64,32,128):
            count=144
            methods=[('Agg',0),('Native',0)]
            for layers in (12,4,8):
                # Alternate order to avoid always placing one axis first.
                axes=('Layer','Token') if int(qps+cap+layers)%4 else ('Token','Layer')
                methods.extend((method,layers) for method in axes)
            for method,layers in methods:
                coarse.append(job(method,layers,count,cap,qps,2026111780))
    odd=[]
    for qps in (3.,5.,7.,9.):
        for method,layers in [('Agg',0),('Native',0)]+[(a,m) for m in (4,8,12) for a in ('Layer','Token')]:
            odd.append(job(method,layers,144,200,qps,2026111780,stage='motivation_odd'))
    return coarse+odd


def qualify_counts():
    initial=ROOT/'analysis/qualification_reserved_N8.json'
    until=time.time()+3600
    while not initial.exists():
        assert time.time()<until,'N8 qualification is not complete. Do not start performance.'
        time.sleep(15)
    assert json.loads(initial.read_text())['passed']
    for count in (32,128):
        path=ROOT/'analysis'/f'qualification_reserved_N{count}.json'
        if path.exists():
            assert json.loads(path.read_text())['passed']
            continue
        command=[str(PYTHON),'-u','-m','poc_flashvep.dense_ar_matched_axis_pd.qualification',
                 '--count',str(count),'--native-version','10','--candidate-version','10']
        with (ROOT/'logs'/f'qualification_N{count}_controller.log').open('a') as log:
            result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=10800)
        assert result.returncode==0, 'State qualification failed: '+str(count)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--qualify',action='store_true')
    p.add_argument('--matrix',action='store_true')
    p.add_argument('--probe',action='store_true')
    args=p.parse_args()
    initialize();pools();historical_outputs()
    try:
        if args.qualify:
            qualify_counts()
        if args.probe:
            assert json.loads((ROOT/'analysis/qualification_reserved_N128.json').read_text())['passed']
            for cap in (16,200):
                for method in ('Native','Layer','Token','Agg'):
                    cfg=job(method,12 if method in ('Layer','Token') else 0,
                        72,cap,6.,2026111781,stage='probe')
                    run_job(cfg)
            write(ROOT/'state/probe_complete.json',dict(time=time.time(),passed=True))
        if args.matrix:
            assert json.loads((ROOT/'state/probe_complete.json').read_text())['passed']
            configs=matrix()
            write(ROOT/'state/matrix_manifest.json',dict(time=time.time(),
                expected=[c['name'] for c in configs],count=len(configs),
                primary_BS=[8192,1024],common_D_admission_cap=8,
                scope='Fixed partitions. Discovery has one independent start per point. Every output uses the same N144 prompt pool and trace. Output200 has all nine QPS points. Confirmation is a separate matrix.'))
            for i,cfg in enumerate(configs):
                run_job(cfg)
                write(ROOT/'state/matrix_progress.json',dict(time=time.time(),complete=i+1,total=len(configs),last=cfg['name']))
            write(ROOT/'state/discovery_complete.json',dict(time=time.time(),count=len(configs)))
    except BaseException as error:
        write(ROOT/'state/controller_stopped.json',dict(time=time.time(),error=repr(error),
            scope='No performance gate is inferred from an incomplete controller. Preserve logs and resume after diagnosis.'))
        raise
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()

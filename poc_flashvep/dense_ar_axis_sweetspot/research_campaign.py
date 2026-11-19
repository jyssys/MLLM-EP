"""Run finite discriminating controls, then independent confirmation."""
import fcntl
import json
import os
import shutil
import time
from .common import ROOT,PREVIOUS,QUIET_LOCK,DEADLINE,write,save,bind_legacy,preserve,resources,digest
from .configuration import base,state
from .run import execute


def freeze(label,configs,reason):
    path=ROOT/'state'/(label+'_design.json')
    value=dict(names=[c['name'] for c in configs],reason=reason,
        source_scope='Frozen inference from 4cf3f9. Fixed partitions. No dynamic scheduler.',
        deadline=DEADLINE,configs={c['name']:c for c in configs})
    if path.exists():
        assert json.loads(path.read_text())==value
    else:
        write(path,value)
    for cfg in configs:
        save(cfg)
    return configs


def performance(method,m,output,qps,seed,batch=1,window=32,count=72,stage='mechanism'):
    name=f'{stage}_{method}_Last{m}_N{count}_O{output}_q{qps:g}_s{seed}_B{batch}_W{window}_v1'
    return base(name,method,m,count,output,qps,seed,batch,window,
        phase='AXIS_SWEETSPOT_'+stage.upper(),matrix_stage=stage)


def run(label,configs):
    from .online_audit import audit
    for i,cfg in enumerate(configs):
        if time.time()>DEADLINE-3600:
            write(ROOT/'state/research_budget_stop.json',dict(time=time.time(),stage=label,
                next=cfg['name'],status='NOT_RUN',reason='Keep one hour for analysis and publication.'))
            return False
        write(ROOT/'state/research_progress.json',dict(time=time.time(),stage=label,
            completed=i,total=len(configs),name=cfg['name'],status='RUNNING'))
        assert execute(cfg['name']),cfg['name']
        audit(cfg['name'])
        print('RESEARCH_PROGRESS',label,i+1,len(configs),cfg['name'],flush=True)
    write(ROOT/'state'/(label+'_complete.json'),dict(time=time.time(),starts=len(configs)))
    return True


def main():
    bind_legacy();preserve()
    # A queued controller must not seize an active or failed initial campaign.
    initial=json.loads((ROOT/'state/initial_controller_v2.json').read_text())
    while not (ROOT/'state/initial_complete.json').exists():
        assert not (ROOT/'state/initial_stopped_v2.json').exists(), 'Initial correctness/calibration failed'
        assert time.time()<DEADLINE-3600
        try:
            os.kill(initial['pid'],0)
        except ProcessLookupError:
            raise RuntimeError('Initial controller exited without a completion gate')
        time.sleep(10)
    from ..dense_ar_matched_axis_pd.qualification import gate
    try:
        states=[state(a,m,n) for n in (8,32) for m in (4,8,12) for a in ('Layer','Token')]
        freeze('expanded_state',states,'Fresh N8 then N32 full-state gates. Last2 and Native were qualified by the initial campaign.')
        for c in states:
            assert execute(c['name']),c['name']
            gate(c['name'],f'state_Native_Last0_N{c["count"]}_B1_v2')
            print('EXPANDED_STATE_PASS',c['name'],flush=True)
        write(ROOT/'state/expanded_state_complete.json',dict(time=time.time(),passed=True,names=[c['name'] for c in states]))
        adapter='fixed_cohort2_v3_qualification_design.json'
        shutil.copyfile(PREVIOUS/'state'/adapter,ROOT/'state'/adapter)
        write(ROOT/'state/reused_batch2_state_contract.json',dict(time=time.time(),
            evidence='CORRECTNESS_ONLY',source=str(PREVIOUS/'state'/adapter),SHA256=digest(ROOT/'state'/adapter),
            scope='Frozen prep1/Decode2 v3 adapter, originally N18 full-state qualified in the base. New WIP controls keep the same adapter. Actual geometry/output/query/credit gates run again for every new performance start.'))
        wip=[performance(a,8,16,6.,2026111901,2,w,72,'source_window_control')
            for w,axes in ((32,('Layer','Token')),(4,('Token','Layer'))) for a in axes]
        freeze('source_window',wip,'Only source outstanding window changes32 to4. H1 service difference versus guard/queue-location hypotheses predict different wait decomposition. Same prep1/Decode2 and partitions.')
        if not run('source_window',wip):
            return
        arms=[('Native',0)]+[(a,m) for m in (2,4,8,12) for a in ('Layer','Token')]
        mechanism=[]
        for index,q in enumerate((2.,6.)):
            ordered=arms[index:]+arms[:index]
            mechanism.extend(performance(a,m,16,q,2026111902,stage='mechanism_discovery') for a,m in ordered)
        freeze('mechanism_discovery',mechanism,'Compare effective service and queue amplification at low/high offered load. Exact rows and actualBatch1. Low-load q0.5 calibration is separate.')
        if not run('mechanism_discovery',mechanism):
            return
        long=[]
        for cap in (64,200):
            for q in (2.,6.):
                long.extend(performance(a,m,cap,q,2026111903,count=36,stage='residency_discovery')
                    for a,m in [('Native',0)]+[(a,m) for m in (4,8,12) for a in ('Layer','Token')])
        freeze('residency_discovery',long,'Change Output while keeping action partition, Batch1, resource and SLO fixed. N36 is exact-row discovery, not sustainable capacity. Last2 long-output remains conditional.')
        if not run('residency_discovery',long):
            return
        confirmation=[]
        # Do not select favorable cells after observing the discovery winner.
        for rep,seed in enumerate((2026111920,2026111921,2026111922)):
            ordered=arms[rep:]+arms[:rep]
            for q in (2.,6.):
                confirmation.extend(performance(a,m,16,q,seed,count=72,stage='heldout_confirmation') for a,m in ordered)
        freeze('heldout_confirmation',confirmation,'Three independent server starts/seeds per action and QPS. Prompts are reused controlled originals. These are new arrival seeds, not a held-out prompt distribution.')
        if not run('heldout_confirmation',confirmation):
            return
        write(ROOT/'state/research_GPU_complete.json',dict(time=time.time(),
            scope='Finite planned mechanism controls complete. No online dynamic policy or sustainable-capacity claim.'))
    except BaseException as error:
        write(ROOT/'state/research_stopped.json',dict(time=time.time(),error=repr(error)))
        raise
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            owner=resources()
            if not owner.processes():
                owner.restore(list(range(8)))
        preserve()


if __name__=='__main__':
    main()

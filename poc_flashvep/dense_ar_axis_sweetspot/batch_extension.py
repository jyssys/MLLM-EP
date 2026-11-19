"""Test a prespecified Decode2-to4 discriminator at an unchanged Last8 cut."""
import fcntl
import json
import time
from .common import ROOT,QUIET_LOCK,DEADLINE,bind_legacy,preserve,write,save,resources
from .configuration import state
from .research_campaign import performance,freeze,run
from .run import execute


def designs():
    states=[]
    for count in (8,32):
        for a,m in (('Native',0),('Layer',8),('Token',8)):
            cfg=state(a,m,count)
            cfg.update(name=f'state_{a}_Last{m}_N{count}_B4_v1',D_admission_batch=4,
                matched_D_cohort_size=4,cohort_runtime_version=3,
                entry_module='poc_flashvep.dense_ar_matched_axis_pd.fixed_matched_cohort_v3')
            states.append(cfg)
    performance_configs=[]
    for rep,seed in enumerate((2026111930,2026111931,2026111932)):
        # Alternate the order. Every paired arm uses the same exact arrival trace.
        for batch in ((2,4) if rep%2==0 else (4,2)):
            arms=[('Native',0),('Layer',8),('Token',8)]
            arms=arms[rep:]+arms[:rep]
            performance_configs.extend(performance(a,m,16,6.,seed,batch=batch,count=72,
                stage='decode_batch_counterfactual') for a,m in arms)
    return states,performance_configs


def main():
    bind_legacy();preserve()
    states,configs=designs()
    freeze('decode4_state',states,'New Batch4 must pass N8 then N32 full-state validation. Reuse the unchanged generic v3 adapter. Keep prep1 and B8192/S1024.')
    freeze('decode_batch_counterfactual',configs,
        'Competing predictions: isolated prefill-GEMM-only ranking is unchanged by Decode batch. Effective P/D balance predicts that Last8 can switch from D-bound atDecode2 to P-bound atDecode4. Change only matched Decode batch on both axes. Three paired fresh arrival seeds. This was declared before the new Source-window or Batch4 outcomes.')
    while not (ROOT/'state/mechanism_discovery_complete.json').exists():
        assert not (ROOT/'state/research_stopped.json').exists()
        assert time.time()<DEADLINE-3600
        time.sleep(10)
    from ..dense_ar_matched_axis_pd.qualification import gate
    try:
        for cfg in states:
            assert execute(cfg['name']),cfg['name']
            gate(cfg['name'],f'state_Native_Last0_N{cfg["count"]}_B4_v1')
            print('BATCH4_STATE_PASS',cfg['name'],flush=True)
        write(ROOT/'state/batch4_state_complete.json',dict(time=time.time(),passed=True,names=[c['name'] for c in states]))
        if run('decode_batch_counterfactual',configs):
            write(ROOT/'state/batch_extension_complete.json',dict(time=time.time(),starts=len(configs)))
    except BaseException as error:
        write(ROOT/'state/batch_extension_stopped.json',dict(time=time.time(),error=repr(error)))
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

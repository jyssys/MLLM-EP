"""Qualify a fixed Decode2 control before any performance comparison."""
import fcntl
import json
import time
from .common import ROOT, QUIET_LOCK, save, write, resources
from .batch_matched import ARMS, state_configuration
from .qualification import gate
from .run import execute


def configuration(method, layers):
    cfg = state_configuration(method, layers)
    cfg.update(name=f'fixed_cohort2_state_{method}_Last{layers}_N18_v1',
        phase='FIXED_COHORT2_CORRECTNESS_PROTOTYPE', matrix_stage='fixed_cohort2_state',
        D_admission_batch=2, matched_D_cohort_size=2,
        entry_module='poc_flashvep.dense_ar_matched_axis_pd.fixed_matched_cohort',
        fixed_cohort_prototype=True,
        timeout=480, wall_timeout=600,
        matched_request_batches=dict(P_prefill=1,D_prefill=1,D_decode=2),
        first_token_release='AFTER_ALL_COHORT_FIRST_NOTICES_EXIST_WITH_ORIGINAL_GENERATION_TIMESTAMPS')
    cfg.pop('matched_request_batch',None)
    return cfg


def main():
    configs = [configuration(a,m) for a,m in ARMS]
    design = ROOT/'state/fixed_cohort2_qualification_design.json'
    value = dict(names=[c['name'] for c in configs], count=18,
        actual_batches=dict(P_prefill=1,D_prefill=1,D_decode=2),
        B=8192,S=1024,phase='CORRECTNESS_ONLY_NOT_ONLINE_PERFORMANCE',
        scope='Fixed FIFO cohorts, not a Dynamic Cut scheduler. Whole-prompt suffix requests prepare in separate legal B8192 iterations. Decode starts after all cohort prompt work is submitted. The first-token barrier preserves generation timestamps and puts collection cost in client TTFT. Native keeps its original cached full-cohort Decode and P-token reuse path. Do not claim TTFT/ITL gains from this correctness run. Stop this prototype on a State/Output/Lifetime error. Batch1 has a separate unchanged qualified inference path.')
    if design.exists():
        assert {k:v for k,v in json.loads(design.read_text()).items() if k!='epoch'} == value
    else:
        write(design,dict(epoch=time.time(),**value))
    try:
        results=[]
        for cfg in configs:
            assert execute(save(cfg)),cfg['name']
            results.append(gate(cfg['name'],configs[0]['name']))
            write(ROOT/'state/fixed_cohort2_qualification_progress.json',dict(epoch=time.time(),
                completed=len(results),total=len(configs),last=cfg['name']))
        write(ROOT/'state/fixed_cohort2_qualification_complete.json',dict(epoch=time.time(),
            passed=True,names=[r['name'] for r in results],evidence='CORRECTNESS_ONLY',
            scope='Output, full prompt KV, hidden and layer visitation passed. Actual steady Decode2 geometry and first-token holds still require an explicit input/log audit before performance.'))
    except BaseException as error:
        write(ROOT/'state/fixed_cohort2_qualification_stopped.json',dict(epoch=time.time(),
            error=repr(error),evidence='FAILED',scope='Confine the verdict to this explicit fixed-cohort prototype. Do not qualify its performance or change any frozen Batch1 configuration.'))
        raise
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            owner=resources()
            if not owner.processes():
                owner.restore(list(range(8)))


if __name__=='__main__':
    main()

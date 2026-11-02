"""Recover a source-guard startup failure without changing acceptance or raw data."""
import copy
from .common import *
from .initial_campaign import run
from .primary_campaign import freeze_slo,subset_parity
from .search_plan import make


def main():
    record=copy.deepcopy(next(r for r in make()['primaries'] if r['model']=='huginn'))
    source=record['calibration'][0]
    cfg=json.loads((ROOT/'configs'/(source+'.json')).read_text())
    cfg['name']=source+'_source_guard_retry_v2'
    target=ROOT/'configs'/(cfg['name']+'.json')
    if target.exists():assert json.loads(target.read_text())==cfg
    else:write(target,cfg)
    write(ROOT/'raw'/source/'failure_annotation.json',dict(time=time.time(),classification='SOURCE_HASH_STARTUP_GUARD',
        failed_before_model_execution=True,measured_requests=0,
        reason='run.py changed while fresh native interpreters verified startup hashes. The guard correctly rejected the start. Preserve the failed log. Use a new namespace and an edit lock for later source changes.'))
    write(ROOT/'state/huginn_primary_recovery.json',dict(stage='CALIBRATION_RETRY',name=cfg['name'],time=time.time()))
    assert run(cfg['name']),'Retry calibration did not complete'
    record['calibration'][0]=cfg['name']
    assert freeze_slo(record),'Both 16-request calibration cohorts are required'
    ledger=[];reference=None
    for name in record['pilots']:
        write(ROOT/'state/huginn_primary_recovery.json',dict(stage='HELDOUT_QPS',name=name,time=time.time()))
        okay=run(name) and subset_parity(record['gate'][0],name)
        if okay and reference:okay=subset_parity(reference,name)
        if reference is None:reference=name
        if okay:
            configure()
            from ..looped_pd_atomic_prefill import evidence
            evidence.ROOT=ROOT
            okay=evidence.full_work(name)['passed']
        ledger.append(dict(name=name,passed=okay))
        write(ROOT/'state/huginn_primary_recovery_results.json',ledger)
        if not okay:break
    write(ROOT/'state/huginn_primary_recovery.json',dict(stage='HUGINN_PRIMARY_RECOVERED_REVIEW_REQUIRED',time=time.time(),results=ledger))
    restore_if_idle()


if __name__=='__main__':main()

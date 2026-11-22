"""Cancel only never-started stale-map jobs and retain explicit NOT_RUN receipts."""
import json
import time
from .common import ROOT,write,digest
from .work_contract import validate

def main():
    plan=json.loads((ROOT/'configs/c3_family_qualification_v4_schedule_fix1.json').read_text());rows=[]
    assert (ROOT/'configs/c3_stock_geometry_repair_v6.json').exists()
    for item in plan['items']:
        if item['arm']!='StockToken':continue
        final=json.loads((ROOT/'configs'/(item['jobs'][-1]+'.json')).read_text())
        try:validate(final,json.loads((ROOT/'data'/final['dataset']).read_text()))
        except AssertionError as error:failure=repr(error)
        else:raise AssertionError('Do not cancel a valid candidate')
        for name in [item['teacher'],item['strict']]+item['jobs']:
            folder=ROOT/'raw'/name
            if folder.exists():
                rows.append(dict(name=name,status='ALREADY_STARTED_OR_RECORDED_UNTOUCHED'));continue
            cfg_path=ROOT/'configs'/(name+'.json');cfg=json.loads(cfg_path.read_text());folder.mkdir()
            write(folder/'configuration.json',cfg)
            write(folder/'preflight_cancellation.json',dict(time=time.time(),status='NOT_RUN',launched=False,
                reason='SUPERSEDED_STALE_REQUEST_PARTITION_CONFIGURATION',validation_failure=failure,
                replacement_plan='c3_stock_geometry_repair_v6.json',config_SHA256=digest(cfg_path),
                scope='No actor or model was started. Retain the original configuration. The action-local old teacher is also superseded.'))
            write(folder/'status.json',dict(status='NOT_RUN',reason='CONFIGURATION_PREFLIGHT_CANCELLATION',time=time.time()))
            write(folder/'dispatcher_result.json',dict(complete=False,returncode=None,elapsed_seconds=0,launched=False,
                source_SHA256=cfg['source_capsule_SHA256'],status='NOT_RUN',reason='STALE_MAP_PREFLIGHT'))
            rows.append(dict(name=name,status='NOT_RUN',launched=False))
    write(ROOT/'state/stale_stock_predispatch_cancellation_v2.json',dict(time=time.time(),rows=rows,
        scope='Cancel only immutable configurations that fail the actual dataset work contract. No running GPU process is stopped. '
            'The parent qualification failure is a configuration rejection,not a failed GPU experiment.'))
    print('NEVER_STARTED_STALE_JOBS_RECORDED_NOT_RUN',len(rows),flush=True)

if __name__=='__main__':main()

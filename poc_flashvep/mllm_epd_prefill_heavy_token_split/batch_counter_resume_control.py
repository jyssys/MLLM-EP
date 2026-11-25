"""Resume only the two unmeasured V2 baselines after the saved C46 API failure."""
import json
import os
from pathlib import Path
import time

from . import batch_counter_control as previous
from . import token_confirmation_control as confirmation
from .client import load_replay_counts
from .common import ROOT,digest,experiment_deadline,preserve,write
from .metrics import summarize
from .runtime_endpoint_identity import require_free_ports
from .source_snapshot import snapshot

TAG='batch_COUNTER_C46R1'
REMAINING=['AggDP3','E_PD_DP2']
PRIOR_METHODS=['NativeStrong','NativeCommon','Static10','Idle10']


def main():
    preserve();assert time.time()+3600<experiment_deadline()
    controller=json.loads((ROOT/'state/batch_COUNTER_C46_controller.json').read_text())
    assert not Path('/proc',str(controller['pid'])).exists()
    failure=ROOT/'state/batch_COUNTER_C46_failure.json';failed=json.loads(failure.read_text())
    assert failed['status']=='FAILED_EXECUTION' and failed['completed_cohorts']==16
    assert failed['error']=='HTTPError: HTTP Error 500: Internal Server Error'
    results=[];mapping={}
    for method in PRIOR_METHODS:
        tag='batch_COUNTER_C46_B0_'+method;mapping[method]=tag
        qualification=json.loads((ROOT/'analysis'/f'{tag}_counter_qualification.json').read_text())
        assert qualification['method']==method
        for stage in ['warmup','clean_before','counter','clean_after']:
            run=tag+'_'+stage;metric=summarize(run)
            assert metric['primary_execution_valid'] and metric['completed_count']==100
            results.append(dict(method=method,subset='Main100',mode=stage,run=run,metrics=metric))
    for path in (ROOT/'state').glob('server_batch_COUNTER_C46*.json'):
        if path.name.endswith('.source_snapshot.json'):continue
        lease=json.loads(path.read_text());assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists()
    config=ROOT/'configs'/f'{TAG}.json';assert not config.exists()
    archive=snapshot(TAG)
    path,main_rows,heldout=confirmation.workload()
    counts=load_replay_counts(ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl')
    write(config,dict(time=time.time(),status='FROZEN_BEFORE_V2_COUNTER_REPAIR_RESULTS',
        source_archive_SHA256=archive['archive_SHA256'],manifest_SHA256=digest(path),
        previous_failure_SHA256=digest(failure),new_methods=REMAINING,reused_methods=PRIOR_METHODS,
        QPS=2.,GPU_group=[4,5,6],independent_initializations_per_method=1,
        one_factor='Counter supports the installed V2 prepare_inputs returned batch and nested EncoderRunner output.',
        unchanged='Same baseline family, original model/vision/processor, compiler, B/S, KV, admission and replay contract.',
        no_legacy_runner_fallback=True,no_new_performance_method=True,no_repeated_completed_methods=True,
        observer_gate='All five CPC metrics and clean spread <=5%. Failure remains diagnostic.',
        previous_failure_preserved=True,scope='No backfill of old clean N3 batches. No serving or quality GO.'))
    write(ROOT/'state'/f'{TAG}_controller.json',dict(time=time.time(),pid=os.getpid(),status='RUNNING_REMAINING_BASELINES'))
    old=previous.TAG;previous.TAG=TAG
    try:
        for method in REMAINING:
            assert time.time()+600<experiment_deadline()
            require_free_ports([18840,18841,18842,18846,18847,18848,18849])
            results.extend(previous.run_baseline(method,path,main_rows,heldout,counts))
            mapping[method]=f'{TAG}_B0_{method}'
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            reused_cohorts=16,new_completed_cohorts=len(results)-16,
            error=type(error).__name__+': '+str(error),restart_automatically=False))
        raise
    finally:previous.TAG=old
    assert len(results)==24
    write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),status='COMPLETE_COLLECTION',results=results,
        method_tags=mapping,previous_failure_SHA256=digest(failure),
        independent_initializations_per_method=1,task_quality='PENDING',full_GO=False,
        scope='Four completed original C46 methods plus two fresh baseline initializations after the observer-only API correction. The original C46 execution remains FAILED_EXECUTION.'))
    write(ROOT/'state'/f'{TAG}_controller_complete.json',dict(time=time.time(),status='COMPLETE'));preserve()
    print('COUNTER_REMAINING_BASELINES_COMPLETE',flush=True)


if __name__=='__main__':main()

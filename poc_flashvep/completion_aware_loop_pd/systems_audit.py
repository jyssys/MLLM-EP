"""Account for wire bytes, host delays, graph policy, and observed queue samples."""
from collections import Counter
from .common import *
from .audit_old import events
from .analyze import percentile
from .serving_audit import csv_write


def distribution(values):
    return dict(count=len(values),p50=percentile(values,.5),p95=percentile(values,.95),
        p99=percentile(values,.99),maximum=max(values,default=None))


def inspect(name):
    folder=ROOT/'raw'/name
    receipt=folder/'dispatcher_result.json'
    if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
        return dict(name=name,status='NOT_COMPLETE')
    begin=json.loads((folder/'measurement_begin.json').read_text())['time']
    ev=events(folder,begin)
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    by_role={}
    for path in folder.glob('wire_payload_*.json'):
        value=json.loads(path.read_text())
        by_role[value['role']]=value
    source_commits={r['tid']:r for r in ev if r['event']=='SOURCE_COMMIT'}
    committed_seeds={r['seed'] for r in source_commits.values()}
    migrations=[]
    for path in folder.glob('migration_*.jsonl'):
        migrations += [r for r in map(json.loads,path.read_text().splitlines())
            if r.get('time',0)>=begin and r.get('seed') in committed_seeds]
    ack=[]
    for path in folder.glob('lifetime_owner_*.jsonl'):
        for r in map(json.loads,path.read_text().splitlines()):
            if r['event']=='NATIVE_ACK_RECEIVED' and r['tid'] in source_commits:
                ack.append(1000*(r['time']-source_commits[r['tid']]['time']))
    supply=[r for r in ev if r['event']=='SUPPLY_SAMPLE']
    quanta=[r for r in ev if r['event']=='D_QUANTUM_HOST_COMPLETION' and r.get('choice')]
    freeze=[json.loads(p.read_text()) for p in folder.glob('graph_freeze_*.json')]
    memories=[json.loads(p.read_text()) for p in folder.glob('memory_inventory_close_*.json')]
    source_spans=[]
    for path in folder.glob('atomic_spans_*.json'):
        value=json.loads(path.read_text())
        if value['role']=='prefill':source_spans += [r for r in value['prompt_spans'] if r['time']>=begin]
    return dict(name=name,model=cfg['model'],count=cfg['count'],method=cfg['intervention_method'],
        migration_KV_bytes=sum(r['KV_bytes'] for r in migrations),
        migration_hidden_bytes=sum(r['hidden_bytes'] for r in migrations),
        migration_fragments=sum(r['fragments'] for r in migrations),
        migration_descriptors=sum(r['descriptors'] for r in migrations),
        native_wire_payloads=by_role,commit_to_native_ACK_host_ms=distribution(ack),
        quantum_host_submission_ms=distribution([r['submit_host_ms'] for r in quanta if 'submit_host_ms' in r]),
        quantum_host_enclosure_ms=distribution([r['elapsed_host_ms'] for r in quanta]),
        quantum_model_overruns=sum(r['elapsed_host_ms']>r['choice']['cost_ms'] for r in quanta),
        quantum_observed_samples=len(quanta),
        source_local_skip_reasons=dict(Counter(r['reason'] for r in ev if r['event']=='SOURCE_LOCAL_BOUNDARY_SKIP')),
        D_KV_free_blocks=distribution([r['KV_free_blocks'] for r in supply if r['role']=='decode']),
        D_Decode_ready=distribution([r['Decode_ready'] for r in supply if r['role']=='decode']),
        D_active_Decode=distribution([r['active_Decode'] for r in supply if r['role']=='decode']),
        P_executable_ready=distribution([r['executable_P_ready'] for r in supply if r['role']=='prefill']),
        P_prompt_request_span_records=len(source_spans),
        P_prompt_distinct_host_submissions=len({r['time'] for r in source_spans}),
        P_prompt_depth_and_loop_count=dict(Counter(f"{r['depth']}+{r['loop_count']}" for r in source_spans)),
        graph_freeze=freeze,memory_inventory=memories,
        exclusive_GPU_saved_time='NOT_MEASURED',exclusive_D_GPU_added_time='NOT_MEASURED',
        critical_path_saving='NOT_MEASURED',
        scope='Migration bytes include committed offload seeds only. Per-run wire totals also include Native transfers. Queue samples are host observations. Submission and completion intervals are not GPU kernel occupancy.')


def main():
    configure()
    names=json.loads((ROOT/'primary_plan.json').read_text())['jobs']
    names+=json.loads((ROOT/'recovery_plan.json').read_text())['jobs']
    zero=ROOT/'state/zero_confirmation_jobs.json'
    if zero.exists():names+=json.loads(zero.read_text())
    rows=[inspect(name) for name in names]
    write(ROOT/'systems_accounting.json',rows)
    csv_write(ROOT/'systems_accounting.csv',rows)
    print(json.dumps(dict(runs=len(rows),complete=sum(r.get('status')!='NOT_COMPLETE' for r in rows))))


if __name__=='__main__':
    main()

"""Calibrate actual macro and destination shapes after exact transport changes."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .analyze import records,percentile


def build(batch):
    pool=json.loads((ROOT/'data/discovery.json').read_text())
    data=[pool[(repeat*batch+j)%len(pool)] for repeat in range(3)
        for layers in (0,2,4,8,12) for j in range(batch)]
    name=f'coalesced_cost_B{batch}_N{len(data)}_v1';dataset=name+'.json'
    path=ROOT/'data'/dataset
    if path.exists():assert json.loads(path.read_text())==data
    else:write(path,data)
    return base(name,'FORCED_QUALIFICATION',count=len(data),cap=16,request_caps=[2]*len(data),
        dataset=dataset,arrival_offsets=[0.]*len(data),arrival_rate=None,
        forced_layers=[0,2,4,8,12],P_group_size=batch,D_wait_for_source_macro=True,
        hidden_source_slots=32,D_admission_batch=16,calibration_serial=True,
        calibration_serial_group_size=batch,verified_ByteLevel_priming=True,
        direct_phase_norm=True,nixl_manual_coalesce=True,nixl_descriptor_sort=False,
        execution_events=True,clean_timing=False,warmup_batch_sizes=[1,4],
        phase='ISOLATED_EVENT_AND_HOST_CALIBRATION_NOT_SERVING_OR_KERNEL_UNION')


def parse(name):
    folder=ROOT/'raw'/name
    req={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
    handoffs={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    submissions={r['request_id']:r['actual_send_start'] for r in records(folder/'actual_submissions.jsonl')}
    events=json.loads((folder/'CUDA_event_enclosures_decode.json').read_text())['rows']
    result=[]
    for commit in records(folder/'actual_cut_commits.jsonl'):
        ids=commit['requests'];layers=commit['offloaded_layers'];key=set(ids)
        frames=[r for r in events if key.intersection(r.get('requests',[]))]
        first=[]
        for rid in ids:
            row=next((r for r in frames if rid in r['requests']),None)
            assert row is not None
            first.append(row)
        distinct=list({id(row):row for row in first}.values())
        assert all(len(row['requests'])==len(ids) for row in distinct),'Calibration destination shape differs from its lookup key'
        source_ready=max(handoffs[r] for r in ids)
        op_start=min(r['start_monotonic'] for r in distinct)
        op_end=max(r['end_monotonic'] for r in distinct)
        first_output=max(req[r]['first_token'] for r in ids)
        result.append(dict(name=name,requests=ids,layers=layers,batch=len(ids),
            actual_destination_shapes=[len(r['requests']) for r in distinct],
            P_prefix_host_ms=1000*(source_ready-min(submissions[r] for r in ids)),
            handoff_to_D_stream_start_ms=1000*(op_start-source_ready),
            D_suffix_event_ms=max(r['elapsed_ms'] for r in distinct) if layers else 0.,
            D_commit_host_ms=max(0.,1000*(first_output-op_end)) if layers else 0.,
            Native_import_to_first_submit_ms=1000*(first_output-source_ready) if not layers else None,
            source_ready=source_ready,D_stream_start=op_start,D_stream_end=op_end,
            first_output=first_output))
    return result


def model(names):
    target=ROOT/'configs/coalesced_cost_model_v1.json'
    if target.exists():
        assert json.loads(target.read_text())['calibration_names']==names
        return
    rows=[r for name in names for r in parse(name)]
    model=json.loads((ROOT/'configs/revised_cost_model_v3.json').read_text())
    mapping={'P_prefix_ms':'P_prefix_host_ms','handoff_ms':'handoff_to_D_stream_start_ms',
        'D_suffix_ms':'D_suffix_event_ms','D_commit_ms':'D_commit_host_ms'}
    for table,field in mapping.items():
        model[table]={}
        for batch in sorted({r['batch'] for r in rows}):
            model[table][str(batch)]={str(layers):percentile([r[field] for r in rows
                if r['batch']==batch and r['layers']==layers],.95) for layers in (0,2,4,8,12)}
    model['uncertainty'].update(source_bias_ms=0.,source_absolute_p95_ms=55.,
        completion_upper_margin_ms=110.,known_drain_underprediction_p95_ms=70.)
    model.update(calibration_names=names,calibration_epoch=time.time(),
        generic_runtime='EXACT_BYTELEVEL_PRIMING_EXPLICIT_NORM_PHYSICAL_RANGE_COALESCING',
        scope='Small-sample empirical p95 enclosures and Host costs at actual shapes. Not kernel union, WCET, clean-serving cost, or unknown-future safety. Validate completion coverage on separate discovery runs.')
    write(target,model)
    write(ROOT/'analysis/coalesced_primitive_calibration.json',dict(rows=rows,
        sources={name:digest(ROOT/'raw'/name/'source.zip') for name in names},
        evidence='MEASURED',scope=model['scope']))


def main():
    try:
        assert json.loads((ROOT/'analysis/coalesced_all_cuts_full_state_gate.json').read_text())['passed']
        names=[]
        for batch in (1,2,3,4,8,16):
            cfg=build(batch);name=save(cfg);assert execute(name,restore=False)
            assert against_previous(name)['passed'];names.append(name)
            print('COALESCED_COST',name,flush=True)
        model(names)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()

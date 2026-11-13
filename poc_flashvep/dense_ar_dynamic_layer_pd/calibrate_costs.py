"""Calibrate event enclosures and completion paths without an online speedup claim."""
import fcntl,statistics
from .common import *
from .prepare import base
from .qualify import dynamic_config
from .run import execute
from .analyze import records,compare,percentile


def parse(name):
    folder=ROOT/'raw'/name
    commits=records(folder/'actual_cut_commits.jsonl');cuts={rid:r['cut'] for r in commits for rid in r['requests']}
    handoffs={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
    source=json.loads((folder/'CUDA_event_enclosures_prefill.json').read_text())['rows']
    destination=json.loads((folder/'CUDA_event_enclosures_decode.json').read_text())['rows']
    rows=[]
    for frame in source:
        ids=frame['requests']
        if not ids or not all(rid in cuts for rid in ids) or not frame['prefilling']:continue
        layers=36-cuts[ids[0]];assert all(cuts[rid]==36-layers for rid in ids)
        target=next(f for f in destination if f['prefilling'] and set(ids)<=set(f['requests']))
        assert set(target['requests'])==set(ids),'Use a matched batch for primitive calibration'
        source_gap=max(0.,1000*(max(handoffs[rid] for rid in ids)-frame['end_monotonic']))
        if layers:
            transfer_gap=max(0.,1000*(target['start_monotonic']-max(handoffs[rid] for rid in ids)))
            commit_gap=max(0.,1000*(max(requests[rid]['first_token'] for rid in ids)-target['end_monotonic']))
            suffix=target['elapsed_ms']
        else:
            transfer_gap=max(0.,1000*(max(requests[rid]['first_token'] for rid in ids)-max(handoffs[rid] for rid in ids)))
            suffix=commit_gap=0.
        rows.append(dict(name=name,batch=len(ids),layers=layers,P_model_event_ms=frame['elapsed_ms'],
            P_post_model_to_source_ready_ms=source_gap,P_prefix_completion_ms=frame['elapsed_ms']+source_gap,
            handoff_to_model_or_native_first_ms=transfer_gap,D_suffix_event_ms=suffix,D_commit_to_first_ms=commit_gap,
            actual_requests=ids,evidence='MEASURED_DIAGNOSTIC_ENCLOSURES',
            scope='CUDA events are stream enclosures. Handoff and commit are Host intervals. They are not exclusive wire or kernel times. Output2 is a primitive diagnostic, not a Serving result.'))
    return rows


def main():
    initialize();assert json.loads((ROOT/'state/mixed_protocol_qualified.json').read_text())['passed']
    assert json.loads((ROOT/'state/all_candidates_mixed32_qualified.json').read_text())['passed']
    target=ROOT/'configs/calibrated_cost_model.json'
    if target.exists():print('COST_MODEL_ALREADY_FROZEN',digest(target));return
    pool=json.loads((ROOT/'data/calibration.json').read_text())+json.loads((ROOT/'data/discovery.json').read_text())
    assert len(pool)==80 and len({r['prompt_id'] for r in pool})==80
    write(ROOT/'data/cost_pool.json',pool)
    ref=base('cost_reference_full_N80_O2_v1',count=80,cap=2,dataset='cost_pool.json')
    ref.update(arrival_offsets=[0.]*80,arrival_rate=None,kv_layout='HND',hidden_transport=None,
        phase='PRIMITIVE_OUTPUT_REFERENCE_NOT_SERVING')
    values=[];results=[]
    try:
        p=base('capacity_P_HND_lean_B124_v3',count=16,cap=16,dataset='calibration.json')
        p.update(module='capacity',capacity_mode='P_ONLY',gpus=[0],hidden_transport=None,kv_layout='HND',
            jobs=[dict(batch=b,repeats=3) for b in (1,2,4)],phase='MATCHED_HND_INDEPENDENT_PREFILL_CAPACITY')
        if not execute(save(p),restore=False):return
        if not execute(save(ref),restore=False):return
        for batch in (1,2,3,4):
            count=batch*20
            c=dynamic_config(f'cost_mixed_B{batch}_N{count}_O2_v1',count=8)
            c.update(count=count,cap=2,dataset='cost_pool.json',request_caps=[2]*count,arrival_offsets=[0.]*count,
                forced_layers=[0,2,4,8,12],P_group_size=batch,D_admission_batch=batch,
                numerical_native_reference=False,hidden_verify_hash=False,verify_layer_execution=False,
                verify_native_prefix_KV=False,kv_layout='HND',
                execution_events=True,phase='EXACT_MIXED_PRIMITIVE_COST_CALIBRATION_NOT_SERVING',timeout=600)
            name=save(c)
            if not execute(name,restore=False):return
            gate=compare(ref['name'],name,True)
            results.append(dict(name=name,output_gate=gate));write(ROOT/'state/cost_calibration_progress.json',dict(results=results))
            if not gate['passed']:return
            values.extend(parse(name))
            write(ROOT/'analysis/primitive_completion_cost_samples.json',dict(rows=values,evidence='MEASURED_DIAGNOSTIC_ENCLOSURES'))
        model=dict(P_prefix_ms={},handoff_ms={},D_suffix_ms={},D_commit_ms={},sample_counts={},
            negotiation_idle_ms=2.,negotiation_busy_ms=30.,decode_step_p95_ms=35.)
        for batch in (1,2,3,4):
            key=str(batch)
            for table in ('P_prefix_ms','handoff_ms','D_suffix_ms','D_commit_ms','sample_counts'):model[table][key]={}
            for layers in (0,2,4,8,12):
                selected=[r for r in values if r['batch']==batch and r['layers']==layers];assert len(selected)>=4
                model['sample_counts'][key][str(layers)]=len(selected)
                for table,field,q in (('P_prefix_ms','P_prefix_completion_ms',.5),('handoff_ms','handoff_to_model_or_native_first_ms',.5),
                    ('D_suffix_ms','D_suffix_event_ms',.95),('D_commit_ms','D_commit_to_first_ms',.95)):
                    model[table][key][str(layers)]=percentile([r[field] for r in selected],q)
        cohorts=records(ROOT/'raw/capacity_D_graph_HND_lean_all_caps_v3/cohorts.jsonl')
        step_costs={}
        for batch in (1,4,8,16):
            intervals=[]
            for row in cohorts:
                if row['batch']!=batch or row['output_cap'] not in (16,200):continue
                intervals.extend(1000*(b['time']-a['time'])/(b['count']-a['count']) for seq in row['token_events'].values() for a,b in zip(seq,seq[1:]))
            step_costs[str(batch)]=percentile(intervals,.95)
        model['decode_step_by_batch_p95_ms']=step_costs;model['decode_step_p95_ms']=max(step_costs.values())
        negotiation=[r['negotiation_ms'] for batch in (1,2,3,4) for r in records(ROOT/'raw'/f'cost_mixed_B{batch}_N{batch*20}_O2_v1'/'reservation_observations.jsonl')]
        model['negotiation_idle_ms']=percentile(negotiation,.95)
        model['negotiation_busy_ms']=max(model['decode_step_p95_ms'],model['negotiation_idle_ms'])
        model.update(frozen_at=time.time(),calibration_names=[r['name'] for r in results],
            source_data_SHA256=digest(ROOT/'data/cost_pool.json'),future_heldout_information=False,
            scope='Calibrated stream enclosures and Host completion costs. Four samples per shape estimate a diagnostic p95, not a validated p99 bound or an online completion guarantee.')
        write(target,model);write(ROOT/'state/cost_model_freeze_receipt.json',dict(time=time.time(),SHA256=digest(target),SLO_SHA256=digest(ROOT/'configs/frozen_SLO_regimes.json')))
        print('COST_MODEL_FROZEN',digest(target),flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()

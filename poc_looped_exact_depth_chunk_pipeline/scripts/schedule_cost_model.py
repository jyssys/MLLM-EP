"""Separate fresh clean measurements from component-calibrated PP forecasts."""
from collections import defaultdict
import statistics
from poc_looped_exact_depth_chunk_pipeline.contract import *
from poc_looped_exact_depth_chunk_pipeline.dag import schedule
from poc_looped_recurrence_tp_4models.analysis import table,bootstrap


def main():
    requests=[];components=[];cached=[];owners={};validity=[]
    for directory in sorted((ROOT/'results/physics').glob('*')):
        if not (directory/'status.json').exists():continue
        info=read(directory/'status.json');TP=info['tp'];R=info['r'];B=info['chunk']
        owners[(TP,R,B)]=read(directory/'ownership.json')
        for path in sorted(directory.glob('P*.clean.json')):
            row=read(path);values=row['records'];N=len(values[0]['output_ids'])
            requests.append(dict(config=directory.name,TP=TP,R=R,chunk=B,P=row['P'],request_id=row['request_id'],
                repeats=len(values),outputs=N,E2E_ms=statistics.mean(v['wall_ms'] for v in values),
                prefill_ms=statistics.mean((v['metrics']['first_token_ts']-v['metrics']['scheduled_ts'])*1000 for v in values),
                TPOT_ms=statistics.mean((v['metrics']['last_token_ts']-v['metrics']['first_token_ts'])*1000/(N-1) for v in values),
                peak_HBM_bytes=max(r['peak_allocated'] for r in row['memory']),label='MEASURED_CLEAN'))
        for path in sorted(directory.glob('P*.stages.json')):
            row=read(path);valid=abs(row['observer_percent'])<=10
            rank_times=[[r for r in seq if r['rows'] is not None and r['rows']>0] for seq in row['trace']]
            assert all([r['rows'] for r in seq]==[r['rows'] for r in rank_times[0]] for seq in rank_times)
            critical=max(range(TP),key=lambda k:sum(r['duration_ms'] for r in rank_times[k]))
            prefix=0;cached_records=[]
            for index,event in enumerate(rank_times[critical]):
                if prefix<row['P']:
                    components.append(dict(TP=TP,R=R,chunk=B,P=row['P'],request_id=row['request_id'],
                        task_index=index,rows=event['rows'],prefix_before=prefix,compute_ms=event['duration_ms'],
                        cumulative_critical_rank=critical,observer_percent=row['observer_percent'],
                        timing_validity='VALID_COMPONENT' if valid else 'UNVALIDATED',label='MEASURED_COMPONENT_RANK_LOCAL_OUTER'))
                    prefix+=event['rows']
                else:cached_records.append(event['duration_ms'])
            assert prefix==row['P'] and len(cached_records)==3
            cached.append(dict(TP=TP,R=R,chunk=B,P=row['P'],request_id=row['request_id'],
                cached_ms=statistics.mean(cached_records),three_actual_intervals=cached_records,
                timing_validity='VALID_COMPONENT' if valid else 'UNVALIDATED'))
            validity.append(dict(config=directory.name,P=row['P'],request_id=row['request_id'],
                observer_percent=row['observer_percent'],valid=valid))
    table(ROOT/'results/baseline_requests.csv',requests);table(ROOT/'results/measured_stage_costs.csv',components)
    table(ROOT/'results/cached_stage_costs.csv',cached);write(ROOT/'results/observer_validity.json',validity)
    grouped=defaultdict(list)
    for row in requests:grouped[(row['TP'],row['R'],row['chunk'],row['P'])].append(row)
    summary=[]
    for (TP,R,B,P),rows in grouped.items():
        summary.append(dict(TP=TP,R=R,chunk=B,P=P,n_requests=len(rows),outputs=rows[0]['outputs'],
            **{name:bootstrap([r[name] for r in rows]) for name in ['E2E_ms','prefill_ms','TPOT_ms','peak_HBM_bytes']}))
    write(ROOT/'results/baseline_summary.json',summary)
    transfer_path=ROOT/'results/stage_transfers_v1/transfers.json'
    if not transfer_path.exists():
        write(ROOT/'results/schedule_bounds.json',dict(status='NOT_RUN',reason='Physical transfer evidence is not complete.'))
        return
    transfers=read(transfer_path);table(ROOT/'results/measured_transfer_costs.csv',transfers)
    baseline={(r['P'],r['request_id']):r for r in requests if r['TP']==4 and r['R']==4 and r['chunk']==0}
    comp=defaultdict(list)
    for row in components:
        if row['timing_validity']=='VALID_COMPONENT':comp[(row['TP'],row['R'],row['chunk'],row['P'],row['request_id'])].append(row)
    cache={(r['TP'],r['R'],r['chunk'],r['P'],r['request_id']):r for r in cached if r['timing_validity']=='VALID_COMPONENT'}
    forecasts=[];timelines=[]
    for key,rows in comp.items():
        TP,R,B,P,rid=key
        if R!=1 or (P,rid) not in baseline:continue
        native=baseline[(P,rid)];native_key=(4,4,0,P,rid)
        if native_key not in comp or native_key not in cache:continue
        costs=[r['compute_ms'] for r in rows];stages=4//TP
        mode='PP4_edge' if TP==1 else 'PP2_TP2_duplicate_boundary'
        measured=[r for r in transfers if r['rows']==B and r['mode'].startswith(mode)]
        if not measured:continue
        quantum=max(r['median_ms'] for r in measured)
        predicted,tape=schedule(costs,[quantum]*len(costs),stages=stages)
        zero,_=schedule(costs,[0.]*len(costs),stages=stages)
        native_GPU=sum(r['compute_ms'] for r in comp[native_key])
        host_envelope=max(0.,native['prefill_ms']-native_GPU)
        prefill=predicted+host_envelope
        e2e_free=native['E2E_ms']-native['prefill_ms']+prefill
        # This bound grants unchanged native decode and no KV-layout transition.
        # It is not an executable candidate and cannot become measured speedup.
        ideal_e2e=native['E2E_ms']-native['prefill_ms']+zero
        logical_KV=(P+63)*192*16*128*2*2
        point=dict(TP_per_stage=TP,stages=stages,chunk=B,P=P,request_id=rid,
            native_prefill_ms=native['prefill_ms'],native_E2E_ms=native['E2E_ms'],
            modeled_prefill_ms=prefill,zero_transfer_compute_ms=zero,modeled_makespan_ms=predicted,
            native_decode_free_transition_E2E_ms=e2e_free,zero_transfer_free_transition_E2E_ms=ideal_e2e,
            modeled_prefill_reduction_percent=(1-prefill/native['prefill_ms'])*100,
            optimistic_E2E_reduction_percent=(1-e2e_free/native['E2E_ms'])*100,
            zero_transfer_E2E_upper_bound_percent=(1-ideal_e2e/native['E2E_ms'])*100,
            interstage_hidden_bytes=P*2048*2*(3 if TP==1 else 2),
            stage_local_KV_bytes_per_GPU=logical_KV//4,
            native_TP4_transition_remote_KV_lower_bound_bytes=logical_KV*3//4,
            stage_weights_bytes_per_GPU=max(r['unique_parameter_storage_bytes'] for r in owners[(TP,R,B)]),
            busy_utilization=sum(costs)*4/(stages*predicted),
            generic_PP_equal=True,label='CALIBRATED_PP. DECODE_TRANSITION_IS_ORACLE_NOT_ACHIEVABLE',
            caveats='First-depth activation costs projected to other depths. Native envelope residual retained. No integrated PP model validation, contention test, or cache transition.')
        forecasts.append(point)
        timelines.extend(dict(TP_per_stage=TP,P=P,chunk_rows=B,request_id=rid,**r,label='CALIBRATED') for r in tape)
    table(ROOT/'results/calibrated_PP_requests.csv',forecasts);table(ROOT/'results/calibrated_timeline.csv',timelines)
    groups=defaultdict(list)
    for row in forecasts:groups[(row['TP_per_stage'],row['chunk'],row['P'])].append(row)
    modeled=[]
    for (TP,B,P),rows in groups.items():
        modeled.append(dict(TP_per_stage=TP,chunk=B,P=P,n_requests=len(rows),
            prefill_reduction_percent=bootstrap([r['modeled_prefill_reduction_percent'] for r in rows]),
            optimistic_E2E_reduction_percent=bootstrap([r['optimistic_E2E_reduction_percent'] for r in rows]),
            zero_transfer_E2E_upper_bound_percent=bootstrap([r['zero_transfer_E2E_upper_bound_percent'] for r in rows]),
            status='CALIBRATED_NOT_RUNTIME_VALIDATED',generic_PP_equal=True))
    write(ROOT/'results/schedule_bounds.json',dict(status='CALIBRATED_NOT_RUNTIME_VALIDATED' if forecasts else 'NOT_RUN',
        models=modeled,reason=None if forecasts else 'Quality-gated capped physics did not collect one-pass stage sweep.',
        no_measured_pipeline_speedup=True,decode_and_transition='Unchanged native decode with free KV transition is an optimistic upper bound only.'))
    print('Clean request means',len(requests),'component intervals',len(components),'modeled PP requests',len(forecasts))


if __name__=='__main__':main()

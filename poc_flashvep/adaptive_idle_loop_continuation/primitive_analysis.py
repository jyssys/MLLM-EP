"""Analyze serial calibration without treating it as an online slack test."""
import statistics
from collections import defaultdict
from .common import *
from .analyze import pct,table,requests


def summary(values):
    return dict(count=len(values),p50_ms=pct(values,50),p95_ms=pct(values,95),maximum_ms=max(values,default=None))


def analyze(name,reference):
    from ..looped_pd_atomic_prefill.evidence import compare,full_work
    from ..dynamic_loop_split_serving.lifetime_gate import check
    from ..dynamic_loop_split_serving.metrics import summarize
    folder=ROOT/'raw'/name
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    assert cfg['calibration_serial'] and cfg['observe_device_execution']
    receipt=json.loads((folder/'dispatcher_result.json').read_text())
    if not receipt['complete']:return dict(name=name,status='FAILED_DISPATCH')
    outputs=requests(folder);expected={r['seed'] for r in outputs}
    by_role={};operation_rows=[]
    for path in folder.glob('device_execution_*.json'):
        value=json.loads(path.read_text());sums=defaultdict(float)
        for row in value['rows']:
            if not row['measured']:continue
            prompt=[r for r in row['requests'] if r['seed'] in expected and r['generated_tokens']==0 and r['token_count']>1]
            if not prompt:continue
            assert len(prompt)==1,'Serial calibration unexpectedly contains batched prompt work'
            sums[prompt[0]['seed']]+=row['stream_elapsed_ms']
            operation_rows.append(dict(name=name,role=value['role'],seed=prompt[0]['seed'],
                operation=row['operation'],loops=row['loops'],tokens=prompt[0]['token_count'],
                stream_enclosure_ms=row['stream_elapsed_ms']))
        by_role[value['role']]=dict(summary(list(sums.values())),per_request_sum_ms=sums)
    baseline=compare(reference,name);work=full_work(name);lifetime=check(name)
    before=json.loads((folder/'before_measured.json').read_text());after=json.loads((folder/'after_measured.json').read_text())
    counters={}
    for role in ('prefill','decode'):
        key=next(key for key in after if '-'+role+'-' in key)
        counters[role]={field:after[key][field]-before[key][field] for field in
            ('issued_loop_rows','issued_submissions','transport_bytes_sent','transport_completed_count')}
    actual=counters['decode']['issued_loop_rows']-cfg['count']*(cfg['cap']-1)*cfg['depth']
    loops=cfg['depth']-cfg.get('native_partial_cut',cfg['depth'])
    frontier=cfg.get('native_frontier')
    if frontier:loops=cfg['depth']-frontier[0]
    expected_rows=(cfg['count']*(cfg['depth']-frontier[0])*(cfg['length']-frontier[1]) if frontier
        else cfg['count']*cfg['length']*loops)
    metric=summarize(name)
    intervals=json.loads((folder/'service_latency_requests.json').read_text())
    result=dict(name=name,model=cfg['model'],remaining_loops=loops,count=cfg['count'],
        frontier=frontier,D_loop_quantum=cfg.get('idle_D_loop_quantum'),
        output=baseline,full_work=work,lifetime=lifetime,actual_D_prompt_rows=actual,
        expected_D_prompt_rows=expected_rows,streams=by_role,counters=counters,
        state_exactness='NOT_MEASURED_IN_THIS_TIMED_SERIAL_RUN',
        passed=baseline['passed'] and work['passed'] and lifetime['passed'] and actual==expected_rows,
        handoff=summary([r['observed_handoff_ms'] for r in intervals if r['observed_handoff_ms'] is not None]),
        service_TTFT=summary([r['service_TTFT_ms'] for r in intervals]),
        D_prompt_window=summary([r['D_prompt_device_enclosure_ms'] for r in intervals if r['D_prompt_device_enclosure_ms'] is not None]),
        metrics=metric,scope='Measured serial CUDA-event enclosures. Not exclusive kernels, clean serving, simultaneous Decode interference, or a validated safe admission budget.')
    write(folder/'serial_primitive_analysis.json',result)
    return result,operation_rows


def main():
    initialize();results=[];operations=[]
    ledger=json.loads((ROOT/'state/primitive_full.json').read_text())
    for record in ledger['results']:
        if not record['complete']:continue
        result,rows=analyze(record['name'],f'p_{record["model"]}_native_serial8_v1')
        results.append(result);operations.extend(rows)
    write(ROOT/'full_prompt_primitive_costs.json',results)
    table(ROOT/'full_prompt_primitive_costs.csv',[dict(name=r['name'],model=r['model'],loops=r['remaining_loops'],
        passed=r['passed'],P_event_sum_p50_ms=r['streams']['prefill']['p50_ms'],
        D_event_sum_p50_ms=r['streams']['decode']['p50_ms'],D_event_sum_max_ms=r['streams']['decode']['maximum_ms'],
        handoff_p50_ms=r['handoff']['p50_ms'],service_p50_ms=r['service_TTFT']['p50_ms'],
        actual_D_prompt_rows=r['actual_D_prompt_rows'],payload_bytes=r['counters']['prefill']['transport_bytes_sent']/r['count']) for r in results])
    table(ROOT/'full_prompt_operation_enclosures.csv',operations)
    ledger=ROOT/'state/quantum_calibration.json'
    extra=[];extra_operations=[]
    if ledger.exists():
        for record in json.loads(ledger.read_text())['results']:
            if not record.get('complete') or not record['name'].startswith('p_'):continue
            model=json.loads((ROOT/'configs'/(record['name']+'.json')).read_text())['model']
            value,rows=analyze(record['name'],f'p_{model}_native_serial8_v1')
            extra.append(value);extra_operations.extend(rows)
    write(ROOT/'equal_quantum_primitive_costs.json',extra)
    table(ROOT/'equal_quantum_operation_enclosures.csv',extra_operations)
    comparison=[]
    for row in extra:
        from ..native_loop_2d_oracle.dependency import Frontier
        cfg=json.loads((ROOT/'configs'/(row['name']+'.json')).read_text())
        c,k=row['frontier'] if row['frontier'] else (cfg['depth']-row['remaining_loops'],0)
        structural=Frontier(cfg['length'],cfg['depth'],c,k).accounting(row['model'])
        native=next(r for r in results if r['model']==row['model'] and r['remaining_loops']==0)
        baseline=native['streams']['prefill']['per_request_sum_ms']
        measured=row['streams']['prefill']['per_request_sum_ms']
        p_savings=[baseline[k]-value for k,value in measured.items()]
        d_operations=[r['stream_enclosure_ms'] for r in extra_operations if r['name']==row['name'] and r['role']=='decode']
        comparison.append(dict(name=row['name'],model=row['model'],
            family='TOKEN_SUFFIX' if row['frontier'] else 'WHOLE_PROMPT_LOOP',
            D_recurrent_rows_per_request=row['actual_D_prompt_rows']/row['count'],
            structural_D_causal_attention_pairs_per_recurrent_layer=structural['D_causal_pairs_per_recurrent_attention_layer'],
            structural_D_Huginn_Prelude_tokens=structural['D_Huginn_prelude_tokens'],
            structural_D_Huginn_Coda_KV_tokens=structural['D_Huginn_Coda_KV_tokens'],
            P_saved_enclosure_p50_ms=pct(p_savings,50),D_total_enclosure_p50_ms=row['streams']['decode']['p50_ms'],
            D_largest_model_submission_enclosure_ms=max(d_operations),
            payload_GiB=row['counters']['prefill']['transport_bytes_sent']/row['count']/1024**3,
            service_TTFT_p50_ms=row['service_TTFT']['p50_ms'],
            serving_Decode_interference='NOT_MEASURED',
            scope='Eight serial requests in one independent start. Equal recurrent rows are not equal FLOPs. A smaller submission enclosure is not a preemptible kernel or a proven safe online slice.'))
    table(ROOT/'equal_quantum_comparison.csv',comparison)
    print(json.dumps([dict(name=r['name'],passed=r['passed'],P=r['streams']['prefill']['p50_ms'],
        D=r['streams']['decode']['p50_ms'],handoff=r['handoff']['p50_ms']) for r in results]))


if __name__=='__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('serial_suffix_cost_analysis'):main()

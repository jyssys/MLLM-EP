"""Analyze serial CUDA Event enclosures. Do not call them kernel unions."""
import statistics
from collections import defaultdict
from .common import *
from .micro_analysis import csv_write


def quantile(values,p):
    values=sorted(values)
    if not values:return None
    at=(len(values)-1)*p/100;i=int(at);j=min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(at-i)


def distribution(values):
    return dict(count=len(values),mean=statistics.mean(values) if values else None,
        **{f'p{q}':quantile(values,q) for q in (50,95,99)},maximum=max(values,default=None))


def analyze(name):
    configure()
    from ..looped_pd_atomic_prefill.evidence import rows,compare,full_work
    from ..dynamic_loop_split_serving.lifetime_gate import check
    from ..dynamic_loop_split_serving.metrics import summarize
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());folder=ROOT/'raw'/name
    receipt=json.loads((folder/'dispatcher_result.json').read_text())
    if not receipt['complete']:return dict(name=name,status='FAILED_DISPATCH')
    output=rows(name);expected={r['seed'] for r in output};roles={};operations=[]
    for path in folder.glob('device_execution_*.json'):
        value=json.loads(path.read_text());sums=defaultdict(float)
        for row in value['rows']:
            prompt=[r for r in row['requests'] if row['measured'] and r['seed'] in expected and
                r['generated_tokens']==0 and r['token_count']>1]
            if not prompt:continue
            assert len(prompt)==1,'Serial prompt work unexpectedly contains multiple requests'
            r=prompt[0];sums[str(r['seed'])]+=row['stream_elapsed_ms']
            operations.append(dict(name=name,role=value['role'],seed=r['seed'],tokens=r['token_count'],
                depth=r['depth'],loops=row['loops'],ms=row['stream_elapsed_ms'],operation=row['operation']))
        roles[value['role']]=dict(distribution(list(sums.values())),per_request=sums)
    before=json.loads((folder/'before_measured.json').read_text());after=json.loads((folder/'after_measured.json').read_text());counters={}
    for role in ('prefill','decode'):
        key=next(k for k in after if '-'+role+'-' in k)
        counters[role]={field:after[key][field]-before[key][field] for field in
            ('issued_loop_rows','issued_submissions','transport_bytes_sent','transport_completed_count')}
    work=full_work(name);lifetime=check(name)
    reference=cfg.get('serial_output_reference',f"cost_{cfg['model']}_zero_original_serial8_s1")
    parity=compare(reference,name)
    exits=[json.loads(p.read_text()) for p in folder.glob('native_worker_exit_*.json')]
    normal=any(x['passed'] and {r['role'] for r in x['records'] if r['exitcode']==0 and not r['forced_cleanup']}=={'prefill','decode'} for x in exits)
    metric=summarize(name)
    service=json.loads((folder/'service_latency_requests.json').read_text())
    core={}
    for path in folder.glob('core_execution_enclosures_*.json'):
        value=json.loads(path.read_text());components=defaultdict(lambda:defaultdict(float));counts=defaultdict(int)
        for row in value['rows']:
            if row['seed'] not in expected:continue
            components[row['component']][str(row['seed'])]+=row['ms'];counts[row['component']]+=1
        core[value['role']]={kind:dict(distribution(list(values.values())),per_request=dict(values),calls=counts[kind]) for kind,values in components.items()}
    for path in folder.glob('core_graph_execution_*.json'):
        value=json.loads(path.read_text());sums=defaultdict(float);calls=0
        for row in value['rows']:
            if row['seed'] in expected:sums[str(row['seed'])]+=row['ms'];calls+=1
        if sums:core.setdefault(value['role'],{})['packed_recurrent_graph']=dict(distribution(list(sums.values())),per_request=dict(sums),calls=calls)
    actual=counters['decode']['issued_loop_rows']-sum((r['output_tokens']-1)*cfg['depth'] for r in output)
    frontier=cfg.get('native_frontier');cut=cfg.get('native_partial_cut')
    expected_rows=cfg['count']*(cfg['length']-frontier[1])*(cfg['depth']-frontier[0]) if frontier else cfg['count']*cfg['length']*(cfg['depth']-(cfg['depth'] if cut is None else cut))
    result=dict(name=name,model=cfg['model'],policy=cfg['projection_kernel_policy'],frontier=frontier,cut=cut,
        passed=work['passed'] and lifetime['passed'] and parity['passed'] and normal and actual==expected_rows,
        parity=parity,full_work=work,lifetime=lifetime,normal_worker_exit=normal,
        actual_D_prompt_rows=actual,expected_D_prompt_rows=expected_rows,streams=roles,counters=counters,core_component_enclosures=core,
        largest_D_prompt_submission_ms=max((r['ms'] for r in operations if r['role']=='decode'),default=0),
        D_prompt_submissions_per_request=sum(r['role']=='decode' for r in operations)/cfg['count'],
        handoff=distribution([r['observed_handoff_ms'] for r in service if r['observed_handoff_ms'] is not None]),
        service_TTFT=distribution([r['service_TTFT_ms'] for r in service]),metrics=metric,
        payload_bytes_per_request=counters['prefill']['transport_bytes_sent']/cfg['count'],
        scope='Serial instrumented stream enclosures include waits and launch gaps. They are not exclusive kernel time, physical idle, online Decode interference, or serving speedup.')
    write(folder/'serial_cost_analysis.json',result)
    return result,operations


def main():
    initialize();plan=json.loads((ROOT/'state/serial_cost_plan.json').read_text());results=[];ops=[]
    extra=ROOT/'state/core_component_plan.json'
    names=plan['jobs']+(json.loads(extra.read_text())['jobs'] if extra.exists() else [])
    quantum=ROOT/'state/v2_serial_quantum_plan.json'
    if quantum.exists():names+=json.loads(quantum.read_text())['jobs']
    for name in names:
        if not (ROOT/'raw'/name/'dispatcher_result.json').exists():continue
        value=analyze(name)
        if isinstance(value,tuple):r,o=value;results.append(r);ops+=o
        else:results.append(value)
    flat=[]
    for r in results:
        if 'streams' not in r:continue
        original=r['name'].replace('_tuned_','_original_')
        cfg=json.loads((ROOT/'configs'/(r['name']+'.json')).read_text())
        zero_name=cfg.get('serial_baseline',f"cost_{r['model']}_zero_{'tuned' if r['policy']=='tuned_s1' else 'original'}_serial8_s1")
        zero=next((v for v in results if v['name']==zero_name),None)
        p=r['streams']['prefill']['per_request'];base=zero['streams']['prefill']['per_request'] if zero else {}
        saved=[base[k]-v for k,v in p.items() if k in base]
        flat.append(dict(name=r['name'],model=r['model'],policy=r['policy'],passed=r['passed'],
            D_prompt_rows_per_request=r['actual_D_prompt_rows']/len(p),
            P_sum_p50_ms=r['streams']['prefill']['p50'],D_sum_p50_ms=r['streams']['decode']['p50'],
            P_saved_sum_p50_ms=quantile(saved,50),D_largest_submission_ms=r['largest_D_prompt_submission_ms'],
            D_recurrent_graph_sum_p50_ms=r['core_component_enclosures'].get('decode',{}).get('packed_recurrent_graph',{}).get('p50'),
            D_prompt_coda_sum_p50_ms=r['core_component_enclosures'].get('decode',{}).get('prompt_coda',{}).get('p50'),
            D_prelude_sum_p50_ms=r['core_component_enclosures'].get('decode',{}).get('prelude',{}).get('p50'),
            D_submissions_per_request=r['D_prompt_submissions_per_request'],
            service_TTFT_p50_ms=r['service_TTFT']['p50'],handoff_p50_ms=r['handoff']['p50'],
            payload_bytes_per_request=r['payload_bytes_per_request'],
            scope='Measured serial CUDA Event enclosures. P saving is a paired baseline difference, not exclusive kernel saving or an additive E2E decomposition.'))
    write(ROOT/'analysis/serial_path_costs.json',results);csv_write(ROOT/'analysis/serial_path_costs.csv',flat)
    csv_write(ROOT/'analysis/serial_operation_enclosures.csv',ops)
    print(json.dumps(flat))


if __name__=='__main__':main()

"""Analyze marker-free serving without fabricating model-stream start times."""
from collections import defaultdict
from .common import *
from .micro_analysis import csv_write


def analyze(name):
    configure()
    from ..huginn_net_benefit.common import analysis
    from ..looped_pd_atomic_prefill.evidence import full_work,compare,rows
    from ..dynamic_loop_split_serving.lifetime_gate import check
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());folder=ROOT/'raw'/name
    metric=analysis().metrics(name)
    if not metric['complete']:return dict(name=name,complete=False)
    reference=name.replace('_'+cfg['width_arm']+'_','_zero_',1)
    parity=compare(reference,name);work=full_work(name);lifetime=check(name)
    exits=[json.loads(p.read_text()) for p in folder.glob('native_worker_exit_*.json')]
    normal=any(x['passed'] and {r['role'] for r in x['records'] if r['exitcode']==0 and not r['forced_cleanup']}=={'prefill','decode'} for x in exits)
    captures=[json.loads(line) for p in folder.glob('graph_captures_*.jsonl') for line in p.read_text().splitlines()]
    graph_counts={}
    for path in folder.glob('deep_graph_cache_*.json'):
        value=json.loads(path.read_text());c=value['counts'].get('measured',{})
        graph_counts[value['role']]=dict(c,hit_rate=c.get('existing_hits',0)/c['selects'] if c.get('selects') else None,
            actual_query_rows=sorted({r['key'][0] for r in value['geometries'].values() if r.get('measured')}),
            scope='Native packed-prefill graph counters after the measurement marker. Not hardware kernel counters.')
    output=rows(name)
    frontier=cfg.get('native_frontier');cut=cfg.get('native_partial_cut')
    expected_rows=cfg['count']*(cfg['length']-frontier[1])*(cfg['depth']-frontier[0]) if frontier else cfg['count']*cfg['length']*(cfg['depth']-(cfg['depth'] if cut is None else cut))
    metric.update(width_arm=cfg['width_arm'],cap=cfg['cap'],phase=cfg['phase'],parity=parity,
        work=work,lifetime=lifetime,normal_worker_exit=normal,expected_D_prompt_rows=expected_rows,
        exact_output_work_lifetime_passed=parity['passed'] and work['passed'] and lifetime['passed'] and normal and metric['actual_D_prompt_recurrent_rows']==expected_rows,
        measured_graph_captures=sum(bool(r.get('measured_phase')) for r in captures),
        measured_graph_counts=graph_counts,
        Service_TTFT='NOT_MEASURED',P_exclusive_kernel_time='NOT_MEASURED',D_exclusive_kernel_time='NOT_MEASURED',
        scope='Marker-free coordinator delivery latency. Complete finite cohort. Not physical kernel idle, sustainable capacity, or a Service TTFT substitute.')
    write(folder/'width_serving_analysis.json',metric);return metric


def main():
    initialize();results=[]
    for path in sorted((ROOT/'configs').glob('serve_*.json')):
        name=path.stem
        if not (ROOT/'raw'/name/'dispatcher_result.json').exists():continue
        results.append(analyze(name))
    flat=[]
    for r in results:
        if not r['complete']:continue
        zero=next((v for v in results if v.get('model')==r['model'] and v.get('cap')==r['cap'] and
            v.get('count')==r['count'] and v.get('arrival_seed')==r['arrival_seed'] and v.get('width_arm')=='zero'),None)
        fields=('TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms','SLO_goodput','completed_QPS')
        record={k:r.get(k) for k in ('name','model','width_arm','cap','count','arrival_seed','Decode_SLO_screen','exact_output_work_lifetime_passed','offloaded_requests','actual_D_prompt_recurrent_rows','ITL_limit_misses',*fields)}
        if zero:
            for key in fields:
                record[key+'_over_zero']=r[key]/zero[key] if zero[key] else None
            record['native_relative_tail_point_gate']=r['ITL_p99_ms']<=1.05*zero['ITL_p99_ms'] and r['TPOT_p95_ms']<=1.05*zero['TPOT_p95_ms']
        flat.append(record)
    write(ROOT/'analysis/serving_results.json',results);csv_write(ROOT/'analysis/serving_results.csv',flat)
    print(json.dumps(flat))


if __name__=='__main__':main()

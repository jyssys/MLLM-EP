"""Analyze separate CUDA Event enclosures without calling them kernel unions."""
from .common import *
from .analyze import metrics,percentile
from .serving_audit import csv_write


def dist(values):
    return dict(count=len(values),p50_ms=percentile(values,.5),p95_ms=percentile(values,.95),
        p99_ms=percentile(values,.99),maximum_ms=max(values,default=None))


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    measured=metrics(name)
    if not measured.get('complete'):return dict(name=name,status='NOT_COMPLETE'),[]
    begin=json.loads((folder/'measurement_begin.json').read_text())['time']
    traces={}
    for path in folder.glob('sparse_device_*.json'):
        v=json.loads(path.read_text());traces[v['role']]=[r for r in v['rows'] if r['host_submit']>=begin]
    requests=[r for r in map(json.loads,(folder/'requests.jsonl').read_text().splitlines()) if not r.get('warmup')]
    out=[]
    for req in requests:
        source=[r for r in traces.get('prefill',[]) if r['prompt'] and any(
            x['seed']==req['seed'] for x in r['requests'])]
        destination=[r for r in traces.get('decode',[]) if r['prompt'] and any(
            x['seed']==req['seed'] for x in r['requests'])]
        first=min(source,key=lambda r:r['begin']['time']) if source else None
        out.append(dict(name=name,model=cfg['model'],sample_id=req['sample_id'],seed=req['seed'],
            TTFT_ms=req['ttft_ms'],
            CUDA_boundary_to_first_output_ms=1000*(req['first_token']-first['begin']['time']) if first else None,
            arrival_to_CUDA_boundary_ms=1000*(first['begin']['time']-req['arrival']) if first else None,
            boundary_clock_uncertainty_ms=1000*(first['begin']['upper']-first['begin']['lower']) if first else None,
            P_operation_enclosure_sum_ms=sum(r['CUDA_enclosure_ms'] for r in source),
            D_prompt_operation_enclosure_sum_ms=sum(r['CUDA_enclosure_ms'] for r in destination),
            scope='Separate instrumented control. CUDA Event boundary, not an identified first kernel. Shared batch enclosures cannot be added across requests.'))
    reference='earlier_huginn_loop_selection32' if 'eventsx_' in name else f"secondary_{cfg['model']}_loop_selection32"
    clean=metrics(reference)
    ratios={k:measured[k]/clean[k] for k in ('TTFT_p95_ms','ITL_p99_ms','maximum_ITL_ms','completed_QPS','TPOT_p95_ms')}
    summary=dict(name=name,model=cfg['model'],metrics=measured,reference=reference,ratios=ratios,
        five_percent_screen=all(abs(v-1)<=.05 for v in ratios.values()),
        CUDA_boundary_to_first_output=dist([r['CUDA_boundary_to_first_output_ms'] for r in out if r['CUDA_boundary_to_first_output_ms'] is not None]),
        arrival_to_CUDA_boundary=dist([r['arrival_to_CUDA_boundary_ms'] for r in out if r['arrival_to_CUDA_boundary_ms'] is not None]),
        trace=traces,
        scope='GPU-clock execution enclosures. Not exclusive kernels, physical idle, or marker-free Service TTFT.')
    return summary,out


def main():
    configure();summaries=[];requests=[]
    for name in json.loads((ROOT/'state/recovery_event_jobs.json').read_text()):
        s,r=analyze(name);summaries.append(s);requests.extend(r)
    write(ROOT/'recovery_event_diagnostics.json',summaries)
    csv_write(ROOT/'recovery_event_request_boundaries.csv',requests)
    print(json.dumps([{k:v for k,v in s.items() if k not in ('trace','metrics')} for s in summaries]))


if __name__=='__main__':
    main()

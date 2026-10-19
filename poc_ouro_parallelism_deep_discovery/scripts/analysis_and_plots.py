"""Aggregate completed runs without converting component costs into E2E gains."""
import argparse
import collections
import csv
import datetime
import math
import random
import statistics
from poc_ouro_parallelism_deep_discovery.contract import *

def table(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def svg(path,series,title,xlabel,ylabel):
    """Small editable vector chart. Values are already aggregated by evidence lane."""
    from xml.sax.saxutils import escape
    W,H=900,500;left,right,top,bottom=85,205,55,70
    points=[p for s in series for p in s['points']]
    if not points:return
    xmin,xmax=min(p[0] for p in points),max(p[0] for p in points)
    ymin=0;ymax=max(p[1] for p in points)*1.1 or 1
    def X(v):return left+(v-xmin)/(xmax-xmin or 1)*(W-left-right)
    def Y(v):return H-bottom-(v-ymin)/(ymax-ymin)*(H-top-bottom)
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-size="19" font-family="sans-serif">{escape(title)}</text>',
        f'<path d="M{left},{top} V{H-bottom} H{W-right}" fill="none" stroke="#222"/>']
    for i in range(6):
        y=ymax*i/5;py=Y(y)
        lines += [f'<path d="M{left},{py} H{W-right}" stroke="#ddd"/>',
            f'<text x="{left-10}" y="{py+5}" text-anchor="end" font-size="12">{y:.3g}</text>']
    for x in sorted({p[0] for p in points}):
        lines.append(f'<text x="{X(x)}" y="{H-bottom+23}" text-anchor="middle" font-size="12">{x:g}</text>')
    colors=['#164e8e','#b94324','#33825c','#8853a2','#c18c00','#555555']
    for i,s in enumerate(series):
        color=colors[i%len(colors)];pts=sorted(s['points'])
        lines.append(f'<polyline points="{" ".join(f"{X(x):.2f},{Y(y):.2f}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x,y in pts:lines.append(f'<circle cx="{X(x):.2f}" cy="{Y(y):.2f}" r="3" fill="{color}"/>')
        ly=top+22*i
        lines += [f'<path d="M{W-right+15},{ly} h22" stroke="{color}" stroke-width="3"/>',
            f'<text x="{W-right+45}" y="{ly+4}" font-family="sans-serif" font-size="12">{escape(s["name"])}</text>']
    lines += [f'<text x="{left+(W-left-right)/2}" y="{H-15}" text-anchor="middle" font-family="sans-serif">{escape(xlabel)}</text>',
        f'<text x="20" y="{H/2}" transform="rotate(-90 20 {H/2})" text-anchor="middle" font-family="sans-serif">{escape(ylabel)}</text>','</svg>']
    path.write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True)
    p.add_argument('--native-family',choices=['default','async','asyncR1','fused'],default='default',
        help='Keep native options in separate summaries. Never pool different execution configurations.')
    args=p.parse_args()
    output=ROOT/'analysis'/args.label;output.mkdir(parents=True,exist_ok=False)
    native=[];operators=[];paged=[];correction=[];parity=[]
    for run in sorted((ROOT/'outputs').iterdir()):
        guard=run/'guard_status.json'
        if not guard.exists() or not read(guard).get('success'):continue
        prefix={'default':'native_base','async':'control_async_base','asyncR1':'control_asyncR1_base','fused':'control_fused_base'}[args.native_family]
        native_states=run.glob('*_TP*/status.json') if run.name.startswith(prefix) else []
        for status in native_states:
            if not read(status).get('success'):continue
            for file in status.parent.glob('P*_request*_repeat*.json'):
                value=read(file);metrics=value['metrics']
                native.append(dict(run=run.name,native_family=args.native_family,model=value['size'],R=value['R'],P=value['TP'],
                    physical_gpus=','.join(map(str,value['physical_gpus'])),context=value['prompt'],request_id=value['request_id'],
                    repeat=value['repeat'],wall_ms=value['wall_ms'],
                    engine_TTFT_ms=metrics['first_token_latency']*1000 if metrics else None,
                    engine_TPOT_ms=(metrics['last_token_ts']-metrics['first_token_ts'])*1000/(metrics['num_generation_tokens']-1) if metrics else None,
                    generated_tokens=len(value['output_ids']),evidence_class='LIVE_E2E',
                    workload='Controlled teacher-forced hardware work, not task quality.',source=str(file.relative_to(ROOT))))
        file=run/'components.json'
        if file.exists():
            for r in read(file)['results']:
                for arm in ['local','TP','remote_round_trip','replica_split','one_worker_combined']:
                    t=r[arm]
                    operators.append(dict(run=run.name,model=r['model'],P=r['P'],rows=r['rows'],arm=arm,
                        graph=t.get('graph',False),median_stream_ms=t.get('median_critical_stream_ms'),median_host_ms=t['median_ms'],
                        sample_count=len(t.get('critical_stream_ms',t['samples_ms'])),evidence_class='GPU_MICROBENCH',
                        residency='Warm same-operator replay, not whole-model residency.'))
        file=run/'paged_depth.json'
        if file.exists():
            for r in read(file)['records']:
                paged.append(dict(context=r['context'],B=r['B'],page=r['page_size'],
                    contiguous_ms=r['contiguous']['median_ms'],paged_ms=r['paged']['median_ms'],
                    speedup=r['speedup_contiguous_over_paged'],max_abs=r['max_abs'],relative_L2=r['relative_L2'],
                    append_check=r['append_check'],evidence_class='GPU_MICROBENCH',
                    quality='NOT_RUN deterministic API fixture, not live model parity.'))
        file=run/'finite_correction.json'
        if file.exists():
            for r in read(file)['records']:
                correction.append(dict(model=r['size'],request_id=r['id'],method=r['method'],warmup=r['exact_warmup_passes'],
                    rounds=r['correction_round'],last_state_relative_L2=r['state_errors'][-1]['relative_L2'],
                    last_state_bitwise_equal=r['state_errors'][-1]['bitwise_equal'],
                    current_KV_max_abs=r['current_token_KV_max_abs'],logit_KL=r['logit_KL_reference_to_provisional'],
                    top1_agreement=r['greedy_token_agreement'],
                    ideal_core_rounds=r['ideal_P4_core_rounds_without_overhead'],
                    evidence_class='QUALITY_ONLY',scope='Serial numerical probe, not candidate generation quality or performance.'))
    table(output/'native_raw.csv',native);table(output/'operators.csv',operators)
    table(output/'paged_depth.csv',paged);table(output/'finite_correction.csv',correction)
    summaries=[]
    for model,R,P,N in sorted({(r['model'],r['R'],r['P'],r['context']) for r in native}):
        rows=[r for r in native if (r['model'],r['R'],r['P'],r['context'])==(model,R,P,N)]
        summaries.append(dict(model=model,R=R,P=P,context=N,native_family=args.native_family,trials=len(rows),unique_requests=len({r['request_id'] for r in rows}),
            median_wall_ms=statistics.median(r['wall_ms'] for r in rows),
            median_TTFT_ms=statistics.median(r['engine_TTFT_ms'] for r in rows),
            median_TPOT_ms=statistics.median(r['engine_TPOT_ms'] for r in rows),
            uncertainty='Two frozen request shapes, five repeats each. Not a broad independent-request confidence claim.'))
    table(output/'native_summary.csv',summaries)
    for model in MODELS:
        for metric,ylabel in [('median_wall_ms','Controlled request wall (ms)'),('median_TPOT_ms','Engine cached-decode TPOT (ms)')]:
            svg(output/(model+'_'+metric+'.svg'),
                [dict(name=f'TP{P}',points=[(r['context'],r[metric]) for r in summaries if r['model']==model and r['R']==4 and r['P']==P])
                 for P in sorted({r['P'] for r in summaries if r['model']==model})],
                f'{model} fixed R4 native {args.native_family} hardware baseline','Prompt tokens',ylabel)
    for P in [2,4]:
        for model in MODELS:
            rows=[r for r in operators if r['P']==P and r['model']==MODELS[model]['id'] and r['graph']]
            svg(output/f'{model}_P{P}_MLP.svg',[dict(name=arm,points=[(r['rows'],r['median_stream_ms']) for r in rows if r['arm']==arm])
                 for arm in ['local','TP','remote_round_trip']],
                f'{model} P{P} MLP component, warm CUDA Graph replay','Token rows','Critical operator stream duration (ms)')
    for model in MODELS:
        rows=[r for r in correction if r['model']==model and r['method']=='jacobi']
        svg(output/f'{model}_Jacobi_KV_error.svg',
            [dict(name=f'case{i+1} warm{warm}',points=[(r['rounds'],r['current_KV_max_abs']) for r in rows if r['request_id']==request and r['warmup']==warm])
             for i,request in enumerate(sorted({r['request_id'] for r in rows})) for warm in [0,1]],
            f'{model} finite correction, two single-token probes','Correction rounds','Current-token KV max absolute error')
    write(output/'summary.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_family=args.native_family,native=summaries,
        sample_counts=dict(native_trials=len(native),operators=len(operators),paged=len(paged),correction=len(correction)),
        interpretation='Component, task-quality, and live controlled-work evidence are separate. No candidate E2E speedup exists.',
        campaign_completion='Not established by this analysis.'))
    print('ANALYSIS',output,len(native),len(operators),len(paged),len(correction))

if __name__=='__main__':main()

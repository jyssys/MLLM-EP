"""Do not treat gaps inside an existing graph replay as free launch slots."""
import argparse,csv,json,collections
from .common import *
from .analyze import csv_write,summary


def analyze(name):
    initialize();folder=ROOT/'raw'/name
    meta=json.loads((folder/'physical_activity_decode.json').read_text())
    assert not meta.get('quiescent_only'),'A complete graph-gap classification needs full-window tracing'
    whole=json.loads((folder/'physical_interval_analysis.json').read_text())['decode']
    left,right=whole['left'],whole['right'];start,stop=meta['start'],meta['stop']
    scale=(stop['host_midpoint']-start['host_midpoint'])/(stop['cupti_ns']-start['cupti_ns'])
    def host(ns):return start['host_midpoint']+(ns-start['cupti_ns'])*scale
    kernels=[]
    for r in csv.DictReader((folder/'physical_activity_decode.csv').open()):
        if r['kind']!='kernel':continue
        a,b=host(int(r['start_ns'])),host(int(r['end_ns']))
        if b<=left or a>=right:continue
        kernels.append((max(a,left),min(b,right),int(r['graph_id']),int(r['correlation']),int(r['stream'])))
    kernels.sort();observed=json.loads((folder/'discovery_events_decode.json').read_text())
    events=sorted((r for r in observed['rows'] if r.get('gpu_start_host_estimate') is not None),key=lambda r:r['gpu_start_host_estimate'])
    margin=max(.00005,(observed['anchor_host_bracket'][1]-observed['anchor_host_bracket'][0])/2)
    cursor=left;previous=None;gaps=[];event_index=0;active=[]
    for kernel in kernels:
        a,b,graph,correlation,stream=kernel
        if a>cursor:
            while event_index<len(events) and events[event_index]['gpu_start_host_estimate']<=cursor:
                active.append(events[event_index]);event_index+=1
            active=[e for e in active if e['gpu_end_host_estimate']>=cursor]
            enclosing=next((e for e in active if e['gpu_start_host_estimate']+margin<=cursor and a+margin<=e['gpu_end_host_estimate']),None)
            same=bool(previous and graph and correlation and previous[2:]==kernel[2:])
            kind=('INSIDE_ONE_CUDA_GRAPH_REPLAY' if same else
                'INSIDE_ONE_NATIVE_MODEL_ENCLOSURE' if enclosing else
                'OUTSIDE_RECORDED_MODEL_ENCLOSURES' if events else 'NOT_IDENTIFIED')
            gaps.append(dict(start=cursor,end=a,duration_ms=1000*(a-cursor),classification=kind,
                previous_graph=previous[2] if previous else None,next_graph=graph,
                previous_correlation=previous[3] if previous else None,next_correlation=correlation,
                stream=stream,enclosing_operation=enclosing['operation'] if enclosing else None,
                enclosing_loops=enclosing['loops'] if enclosing else None,
                enclosing_batch=len(enclosing['requests']) if enclosing else None))
        if b>cursor:cursor=b;previous=kernel
    if cursor<right:gaps.append(dict(start=cursor,end=right,duration_ms=1000*(right-cursor),classification='RIGHT_BOUNDARY_NOT_IDENTIFIED',
        previous_graph=None,next_graph=None,previous_correlation=None,next_correlation=None,stream=None,
        enclosing_operation=None,enclosing_loops=None,enclosing_batch=None))
    groups={}
    for label in sorted({r['classification'] for r in gaps}):
        values=[r for r in gaps if r['classification']==label]
        groups[label]=dict(count=len(values),seconds=sum(r['duration_ms'] for r in values)/1000,
            durations_ms=summary(r['duration_ms'] for r in values),
            thresholds={str(t):dict(count=sum(r['duration_ms']>=t for r in values),
                seconds=sum(r['duration_ms'] for r in values if r['duration_ms']>=t)/1000) for t in (1,2,5,10,20,30)})
    value=dict(name=name,window_seconds=right-left,total_gap_seconds=sum(r['duration_ms'] for r in gaps)/1000,
        groups=groups,mapping_margin_ms=1000*margin,
        scope='Kernel intervals and recorded graph identities provide the first classification. An enclosure alone does not prove that every later graph was already submitted. A physical gap inside one graph replay is not an ordinary same-stream admission slot. Outside an enclosure does not establish Decode safety, state readiness, or an empty queue.')
    csv_write(folder/'D_gap_context_ge50us.csv',(r for r in gaps if r['duration_ms']>=.05),
        ('start','end','duration_ms','classification','previous_graph','next_graph','previous_correlation','next_correlation','stream','enclosing_operation','enclosing_loops','enclosing_batch'))
    write(folder/'D_gap_context_analysis.json',value)
    print(json.dumps(value));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');analyze(p.parse_args().name)

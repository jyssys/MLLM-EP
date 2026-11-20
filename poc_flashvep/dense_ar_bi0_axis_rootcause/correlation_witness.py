"""Test external IDs on real eager kernels and CUDA graph replays."""
import csv
import json
import os
import sys
import time
from .common import ROOT,write
from .correlated_activity import Activity


def main():
    cfg=json.load(open(sys.argv[1]));folder=ROOT/'raw'/cfg['name']
    import torch
    torch.set_num_threads(1)
    if cfg.get('runtime_only_correlation'):
        from .runtime_only_activity import Activity as SelectedActivity
    elif cfg.get('runtime_api_correlation'):
        from .runtime_activity import Activity as SelectedActivity
    else:
        SelectedActivity=Activity
    collector=SelectedActivity(folder,witness=True)
    x=torch.randn((1024,1024),device='cuda',dtype=torch.bfloat16)
    y=torch.empty_like(x);z=torch.empty_like(x)
    for _ in range(3):torch.mm(x,x,out=y)
    torch.cuda.synchronize()
    stamps=[]
    for token in (101,102):
        collector.push(token);left=collector.stamp()
        for _ in range(3):torch.mm(x,x,out=y)
        torch.cuda.synchronize();right=collector.stamp();collector.pop(token)
        stamps.append(dict(external_id=token,left=left,right=right,mode='EAGER'))
    if cfg.get('runtime_api_correlation') or cfg.get('runtime_only_correlation'):
        for token in (301,302):
            collector.push(token);left=collector.stamp()
            y.add_(x);z.copy_(y);z.mul_(2.)
            torch.cuda.synchronize();right=collector.stamp();collector.pop(token)
            stamps.append(dict(external_id=token,left=left,right=right,mode='RUNTIME'))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.mm(x,x,out=y);torch.mm(y,x,out=z)
    for token in (201,202):
        collector.push(token);left=collector.stamp();graph.replay()
        torch.cuda.synchronize();right=collector.stamp();collector.pop(token)
        stamps.append(dict(external_id=token,left=left,right=right,mode='GRAPH'))
    collector.close()
    mapping={int(r['correlation']):int(r['external_id']) for r in csv.DictReader((folder/'kernels.csv.external.csv').open())}
    kernels=[r for r in csv.DictReader((folder/'kernels.csv').open()) if r['kind']=='kernel']
    checks=[]
    for row in stamps:
        subset=[r for r in kernels if int(r['start_ns'])>=row['left']['cupti_ns'] and int(r['end_ns'])<=row['right']['cupti_ns']]
        checks.append(dict(external_id=row['external_id'],mode=row['mode'],kernels=len(subset),
            mapped=sum(mapping.get(int(r['correlation']))==row['external_id'] for r in subset),
            passed=bool(subset) and all(mapping.get(int(r['correlation']))==row['external_id'] for r in subset)))
    counters=json.loads((folder/'activity_counters.json').read_text())
    result=dict(time=time.time(),checks=checks,eager_correlation_passed=counters['passed'] and all(r['passed'] for r in checks if r['mode']=='EAGER'),
        runtime_correlation_passed=counters['passed'] and all(r['passed'] for r in checks if r['mode']=='RUNTIME'),
        graph_correlation_passed=counters['passed'] and all(r['passed'] for r in checks if r['mode']=='GRAPH'),
        counters=counters,GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),
        evidence='CORRECTNESS_ONLY',scope='Collector ID witness only. No online observer-overhead or serving-performance qualification.')
    write(ROOT/'analysis'/('correlated_collector_witness_'+cfg['name']+'.json'),result)
    destination='runtime_only_correlation_witness.json' if cfg.get('runtime_only_correlation') else 'runtime_api_correlation_witness.json' if cfg.get('runtime_api_correlation') else 'correlated_collector_witness.json'
    write(ROOT/'analysis'/destination,result)
    assert result['eager_correlation_passed'],result
    if cfg.get('runtime_api_correlation') or cfg.get('runtime_only_correlation'):assert result['runtime_correlation_passed'],result
    write(folder/'status.json',dict(status='COMPLETE',time=time.time(),pid=os.getpid()))


if __name__=='__main__':main()

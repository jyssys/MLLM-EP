"""Keep measured enclosures separate from guarded cost-model inputs."""
from collections import defaultdict
from .common import *
from .analyze import percentile


def summarize(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    if not cfg.get('calibration_serial'):return None
    receipt=folder/'dispatcher_result.json'
    if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):return None
    groups=defaultdict(list);sums=defaultdict(lambda:defaultdict(float))
    for file in folder.glob('device_execution_*.json'):
        data=json.loads(file.read_text());role=data['role']
        for row in data['rows']:
            if not row['measured']:continue
            spans=[s for s in row['requests'] if s['token_count']>1 and s['generated_tokens']==0]
            if not spans:continue
            assert len(spans)==1,'This cost model requires serial calibration'
            span=spans[0];terminal=span['depth']+row['loops']==cfg['depth']
            key=(role,span['token_count'],row['loops'],terminal,row['operation'])
            groups[key].append(row['stream_elapsed_ms'])
            sums[role][span['seed']]+=row['stream_elapsed_ms']
    result=dict(name=name,model=cfg['model'],count=cfg['count'],groups=[],per_request_sums=dict(sums),
        scope='Measured serial CUDA Event enclosures. Guarded bounds are model inputs, not certified execution bounds or clean serving effects.')
    for (role,tokens,loops,terminal,operation),values in groups.items():
        p99=percentile(values,.99);guard=max(1.,.2*p99)
        result['groups'].append(dict(role=role,tokens=tokens,loops=loops,terminal=terminal,
            operation=operation,count=len(values),p50_ms=percentile(values,.5),p95_ms=percentile(values,.95),
            p99_ms=p99,maximum_ms=max(values),initial_guard_ms=guard,p99_plus_guard_ms=p99+guard,
            over_p99_plus_guard_count=sum(v>p99+guard for v in values)))
    write(folder/'primitive_cost_summary.json',result)
    return result


def main():
    configure();rows=[]
    for file in sorted((ROOT/'raw').glob('*/device_execution_*.json')):
        if any(r['name']==file.parent.name for r in rows):continue
        result=summarize(file.parent.name)
        if result:rows.append(result)
    write(ROOT/'new_primitive_costs.json',rows)
    print(json.dumps([dict(name=r['name'],D_quanta=sum(g['count'] for g in r['groups'] if g['role']=='decode')) for r in rows]))


if __name__=='__main__':main()

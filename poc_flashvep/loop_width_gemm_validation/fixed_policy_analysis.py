"""Confirm the start-1 operator map without selecting again on later starts."""
from .common import *
from .micro_analysis import csv_write


def main():
    initialize();policy=json.loads((ROOT/'configs/operator_kernel_policy_s1.json').read_text());flat=[];width=[]
    for model in ('huginn','ouro'):
        mapping={(r['role'],r['M'],r['N'],r['K']):r for r in policy['models'][model]['entries']}
        for start in (1,2,3):
            value=json.loads((ROOT/'raw'/f'gemm_{model}_search_s{start}'/'projection_results.json').read_text());grouped={}
            for row in value['results']:
                n,k=row['weight_shape'];choice=mapping[row['role'],row['rows'],n,k];method=choice['method']
                source='reference_graph';exact=True
                if method=='qualified_reference':timing=row['reference_graph']
                elif method=='unpadded':
                    timing=row['unpadded_graph'];source='unpadded_graph'
                    exact=row['unpadded_exact_selection']['equal'] and all(r['equal'] for r in row['unpadded_heldout_exact'])
                else:
                    candidate=next(c for c in row['candidates'] if all(c[key]==v for key,v in choice['algorithm'].items()))
                    source='heldout_graph' if 'heldout_graph' in candidate else 'selection_graph'
                    timing=candidate[source]
                    exact=candidate.get('operator_qualified',False)
                record=dict(model=model,start=start,role=row['role'],M=row['rows'],N=n,K=k,
                    method=method,ms=timing['p50_ms'],reference_ms=row['reference_graph']['p50_ms'],
                    heldout_operator_exact=exact,timing_source=source,
                    scope='Fixed start-1 dispatch. A missing held-out graph uses the explicitly identified selection-graph timing. No later-start reselection.')
                flat.append(record);grouped.setdefault((row['role'],n,k),{})[row['rows']]=(record,row['multiplicity_per_loop'])
            R=32 if model=='huginn' else 4;length=2048 if R==32 else 4096;suffix=length//R
            for batch in (1,2,4,8):
                if any(length not in g or suffix*batch not in g for g in grouped.values()):continue
                loop=sum(g[length][0]['ms']*g[length][1] for g in grouped.values())
                token=sum(g[suffix*batch][0]['ms']*g[suffix*batch][1] for g in grouped.values())*R/batch
                width.append(dict(model=model,start=start,token_batch=batch,loop_dense_sum_ms=loop,
                    token_dense_sum_ms=token,token_over_loop=token/loop,
                    all_heldout_exact=all(g[length][0]['heldout_operator_exact'] and g[suffix*batch][0]['heldout_operator_exact'] for g in grouped.values()),
                    scope='Structural sum of isolated projection timers. Not an executed recurrent core or measured serving speedup. Token batching does not establish ready work or KV availability.'))
    csv_write(ROOT/'analysis/fixed_policy_projection_confirmation.csv',flat)
    csv_write(ROOT/'analysis/fixed_policy_width_confirmation.csv',width)
    write(ROOT/'analysis/fixed_policy_confirmation.json',dict(policy_SHA256=digest(ROOT/'configs/operator_kernel_policy_s1.json'),projections=flat,width=width))
    print(json.dumps(width))


if __name__=='__main__':main()

"""Separate logical causal work from measured serial CUDA enclosures."""
from .common import *
from .analyze import percentile
from .campaign_report import write_csv


def main():
    structural=[];comparisons=[]
    for model,T,R in (('ouro',4096,4),('huginn',2048,32)):
        for c in ((1,2,3) if R==4 else (24,28,30,31)):
            structural.append(dict(model=model,family='loop',cut=c,remaining_loops=R-c,
                recurrent_rows=(R-c)*T,causal_query_key_pairs=(R-c)*T*(T+1)//2,
                scope='Logical causal positions per shared-core attention layer. Not physical GEMM work, measured FLOPs, or time.'))
        for k in sorted({T//4,T//2,3*T//4,T-512,T-256,T-128,T-64,T-32}):
            n=T-k
            structural.append(dict(model=model,family='token',cut=k,suffix_tokens=n,
                recurrent_rows=R*n,causal_query_key_pairs=R*(n*k+n*(n+1)//2),
                scope='Logical causal positions per shared-core attention layer. Prelude, Coda, reinjection, and padding are not included.'))
        loops=3 if model=='ouro' else 8;suffix=3072 if model=='ouro' else 512
        names=[f'pf_{model}_loops{loops}_quantum1_serial8']
        names.extend(p.parent.name for p in (ROOT/'raw').glob(f'pt*_{model}_suffix{suffix}_*/primitive_cost_summary.json'))
        for name in dict.fromkeys(names):
            file=ROOT/'raw'/name/'primitive_cost_summary.json'
            if not file.exists():continue
            value=json.loads(file.read_text());values=list(value['per_request_sums'].get('decode',{}).values())
            groups=[g for g in value['groups'] if g['role']=='decode']
            if not values:continue
            token=name.startswith('pt');pairs=(R*(suffix*(T-suffix)+suffix*(suffix+1)//2) if token else loops*T*(T+1)//2)
            comparisons.append(dict(model=model,name=name,family='token' if token else 'loop',
                matched_recurrent_rows=R*suffix,logical_causal_pairs=pairs,
                D_total_enclosure_p50_ms=percentile(values,.5),D_total_enclosure_p99_ms=percentile(values,.99),
                largest_group_p99_ms=max(g['p99_ms'] for g in groups),
                largest_observed_quantum_ms=max(g['maximum_ms'] for g in groups),
                measured_FLOPs='NOT_MEASURED',
                scope='Serial disjoint-eight calibration. Same recurrent rows do not imply equal Attention, boundary work, or FLOPs. CUDA enclosures are not online safe windows.'))
    write_csv(ROOT/'structural_work_accounting.csv',structural)
    write_csv(ROOT/'matched_rows_primitive_comparison.csv',comparisons)
    write(ROOT/'matched_rows_primitive_comparison.json',comparisons)
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()

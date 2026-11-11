import csv,math,statistics
from .common import *


def csv_write(path,rows):
    if not rows:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def choices(row):
    qualified=[dict(method='qualified_reference',ms=row['reference_graph']['p50_ms'])]
    all_options=[dict(method='unpadded',ms=row['unpadded_graph']['p50_ms'])]
    if row['unpadded_exact_selection']['equal'] and all(x['equal'] for x in row.get('unpadded_heldout_exact',[])):
        qualified+=all_options
    for candidate in row['candidates']:
        if 'heldout_graph' not in candidate:continue
        option=dict(method='cublasLt',ms=candidate['heldout_graph']['p50_ms'],algorithm={key:candidate[key] for key in ('index','origin','id','tile','splitK','reduction','swizzle','stages','workspace_bytes')})
        all_options.append(option)
        if candidate.get('operator_qualified'):qualified.append(option)
    return min(qualified,key=lambda x:x['ms']),min(all_options,key=lambda x:x['ms'])


def main():
    initialize();flat=[];width=[];selections={}
    for folder in sorted((ROOT/'raw').glob('gemm_*')):
        path=folder/'projection_results.json'
        if not path.exists():continue
        value=json.loads(path.read_text());cfg=value['config']
        if cfg.get('smoke'):continue
        grouped={}
        for row in value['results']:
            best,unrestricted=choices(row)
            key=(row['role'],tuple(row['weight_shape']))
            grouped.setdefault(key,{})[row['rows']]=dict(row=row,best=best,unrestricted=unrestricted)
            flat.append(dict(run=folder.name,model=cfg['model'],role=row['role'],module=row['module'],M=row['rows'],
                N=row['weight_shape'][0],K=row['weight_shape'][1],multiplicity=row['multiplicity_per_loop'],
                reference_ms=row['reference_graph']['p50_ms'],best_operator_exact_ms=best['ms'],
                unrestricted_ms=unrestricted['ms'],exact_method=best['method'],
                exact_gain=row['reference_graph']['p50_ms']/best['ms'],
                scope='Isolated projection graph timing. Exact means tested operator inputs, not model state.'))
            selections.setdefault(cfg['model'],{})[str((row['role'],row['rows'],*row['weight_shape']))]=best
        R=32 if cfg['model']=='huginn' else 4;length=2048 if R==32 else 4096
        for loops in (1,2,4,8) if R==32 else (1,2,3):
            chunk=length*loops//R
            for batch in (1,2,4,8):
                rows=chunk*batch
                if any(length not in v or rows not in v for v in grouped.values()):continue
                for method,field in (('qualified_reference',None),('best_operator_exact','best'),('unrestricted_diagnostic','unrestricted')):
                    def cost(c):return c['row']['reference_graph']['p50_ms'] if field is None else c[field]['ms']
                    loop=sum(v[length]['row']['multiplicity_per_loop']*cost(v[length]) for v in grouped.values())*loops
                    token=sum(v[rows]['row']['multiplicity_per_loop']*cost(v[rows]) for v in grouped.values())*R/batch
                    width.append(dict(run=folder.name,model=cfg['model'],method=method,offloaded_loops=loops,
                        token_suffix=chunk,token_batch=batch,loop_dense_sum_ms=loop,token_dense_sum_ms=token,
                        token_over_loop=token/loop,
                        scope='Sum of isolated representative projection timings with structural multiplicities. Not an executed recurrent core, Attention cost, P saving, or serving speedup. Batched Token cases do not prove KV admission or ready-work availability.'))
    csv_write(ROOT/'analysis/projection_kernel_results.csv',flat)
    csv_write(ROOT/'analysis/dense_width_cost_sensitivity.csv',width)
    write(ROOT/'analysis/operator_kernel_choices.json',dict(selections=selections,
        scope='Latest independent micro process per model. Candidate deployment still requires actual model output/KV/Hidden qualification.'))
    write(ROOT/'analysis/micro_summary.json',dict(projections=flat,width=width))
    print('PROJECTION_CELLS',len(flat),'DENSE_SENSITIVITIES',len(width))


if __name__=='__main__':main()

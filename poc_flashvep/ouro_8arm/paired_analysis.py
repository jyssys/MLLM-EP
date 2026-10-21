"""Question-paired accuracy differences. Run variants remain separate."""
import csv
import numpy as np
from .contract import *


def main():
    rows=[json.loads(line) for line in (ROOT/'evaluation_outputs.jsonl').read_text().splitlines()]
    groups={run:{r['id']:r for r in rows if r['run']==run} for run in sorted({r['run'] for r in rows})}
    references=['native_TP1_pilot_v2','native_TP1_pilot_v1','CDB_SD_pilot_v1','quality_SD_pilot_v1','quality_S4_canonical',
        'quality400_S4','quality400_SD','quality400_native_TP1']
    results=[]
    for run,values in groups.items():
        for ref in references:
            if ref not in groups or ref==run:continue
            for domain in ['gsm8k','math500']:
                candidates=sorted(k for k in values.keys() & groups[ref].keys() if values[k]['dataset']==domain)
                shared=[k for k in candidates if values[k].get('prompt_ID_parity',True) and groups[ref][k].get('prompt_ID_parity',True)]
                if not shared:continue
                a=np.array([values[k]['correct'] for k in shared],dtype=float)
                b=np.array([groups[ref][k]['correct'] for k in shared],dtype=float)
                delta=a-b;rng=np.random.default_rng(20261021)
                means=delta[rng.integers(0,len(shared),(5000,len(shared)))].mean(axis=1)*100
                output_same=depth_same=depth_comparable=0
                for k in shared:
                    x=read(ROOT/values[k]['source']);y=read(ROOT/groups[ref][k]['source'])
                    output_same+=x['output_ids']==y['output_ids']
                    if 'depths' in x and 'depths' in y:
                        depth_comparable+=1;depth_same+=x['depths']==y['depths']
                results.append(dict(run=run,reference=ref,dataset=domain,N_common=len(shared),excluded_input_mismatches=len(candidates)-len(shared),correct=int(a.sum()),reference_correct=int(b.sum()),
                    delta_accuracy_pp=float(delta.mean()*100),CI95_low_pp=float(np.quantile(means,.025)),CI95_high_pp=float(np.quantile(means,.975)),
                    correct_to_wrong=int(((b==1)&(a==0)).sum()),wrong_to_correct=int(((b==0)&(a==1)).sum()),
                    token_exact_requests=output_same,depth_trace_comparable_requests=depth_comparable,
                    depth_exact_requests=depth_same if depth_comparable else None,scope='Question-paired bootstrap, 5000 draws, seed 20261021. Not equivalence proof.'))
    if results:
        with (ROOT/'paired_accuracy_deltas.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(results[0]),lineterminator='\n');w.writeheader();w.writerows(results)
    print('PAIRED_COMPARISONS',len(results))


if __name__=='__main__':main()

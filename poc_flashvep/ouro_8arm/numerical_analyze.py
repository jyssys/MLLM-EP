"""Compare last-token component seams without relabeling the historical gate."""
import csv
import torch
from .contract import *
from .numerical_trace import errors

def main():
    torch.set_num_threads(1);summary=[]
    with (ROOT/'first_divergence.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=['pair','label','step','loop','layer','component','max_abs','MAE','relative_L2'],lineterminator='\n');writer.writeheader()
        for r in read(ROOT/'numerical/hf_manifest.json'):
            label=r['label'];values={name:torch.load(ROOT/'.tensor_cache'/f'{name}_{label}.pt',weights_only=False,map_location='cpu')
                for name in ['HF','Unsharded','Custom_TP4','vLLM_TP1','vLLM_TP4'] if (ROOT/'.tensor_cache'/f'{name}_{label}.pt').exists()}
            for a,b in [('HF','Unsharded'),('HF','Custom_TP4'),('HF','vLLM_TP1'),('HF','vLLM_TP4'),('vLLM_TP1','vLLM_TP4')]:
                if a not in values or b not in values:continue
                rows=[]
                for key,x in values[a]['frames'].items():
                    step,loop,layer,op=key.split(':');e=errors(values[b]['frames'][key],x)
                    row=dict(pair=a+' vs '+b,label=label,step=int(step),loop=int(loop)+1,layer=int(layer)+1,component=op,**e)
                    writer.writerow(row);rows.append(row)
                bad=[row for row in rows if row['relative_L2']>.03]
                agreement=[]
                for z,y in zip(values[a]['logits'],values[b]['logits']):agreement.append(int(z.reshape(-1).argmax())==int(y.reshape(-1).argmax()))
                summary.append(dict(pair=a+' vs '+b,label=label,P=r['P'],O=r['O'],
                    first_exceeding_historical_diagnostic_level=bad[0] if bad else None,
                    max_relative_L2=max(v['relative_L2'] for v in rows),argmax_equal=sum(agreement),argmax_total=len(agreement),
                    scope='Aligned last-token-only components. Native layer outputs materialize fused residual for observation.'))
    write(ROOT/'numerical_gate.json',dict(HISTORICAL_STRICT_C2='FAIL_UNCHANGED',prospective='NUMERICALLY_UNQUALIFIED_EXPLORATORY',
        reason='No new numeric threshold is calibrated on final task labels.',comparisons=summary))
    print('NUMERICAL_ANALYZED',len(summary),flush=True)

if __name__=='__main__':main()

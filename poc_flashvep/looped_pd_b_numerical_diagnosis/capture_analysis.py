"""Compare saved boundary snapshots without changing a running GPU job."""
import torch
from .common import ROOT,write
from .linear_probe import metrics


def load(name):
    result={}
    for path in sorted((ROOT/'raw'/name).glob('boundary_operands_*.pt')):
        value=torch.load(path,map_location='cpu',weights_only=True)
        for row in value['records']:
            key=(row['step'],row['depth'],row['operation'])
            if key in result:raise ValueError('Boundary key is not unique. Preserve the ambiguity instead of pairing it silently.')
            result[key]=row
    return result


def main():
    torch.set_num_threads(1)
    names=['ouro_old_canonical_sample233_capture_PD_G1_0','ouro_old_canonical_sample233_capture_B_1']
    a,b=map(load,names);result=[]
    for key in sorted(a.keys()&b.keys()):
        row=dict(step=key[0],depth=key[1],operation=key[2],G1_rows=a[key]['rows'],B_rows=b[key]['rows'])
        for field in ('input_hidden','output_hidden','logits'):
            x,y=a[key][field],b[key][field]
            row[field]=metrics(x,y) if x is not None and y is not None else dict(both_absent=x is None and y is None)
        if a[key]['logits'] is not None:
            row['G1_argmax']=int(a[key]['logits'].argmax());row['B_argmax']=int(b[key]['logits'].argmax())
        result.append(row)
    write(ROOT/'configs/actual_PD_boundary_comparison.json',dict(runs=names,matched_keys=len(result),
         unmatched_left=[list(k) for k in a.keys()-b.keys()],unmatched_right=[list(k) for k in b.keys()-a.keys()],
         comparisons=result,scope='Late instrumented real PD snapshots through output4. Instrumentation can alter scheduling. Not the clean400-request failure.'))
    print(dict(matched=len(result),different=[r for r in result if any(r[k].get('exact') is False for k in ('input_hidden','output_hidden','logits'))][:3]))


if __name__=='__main__':main()

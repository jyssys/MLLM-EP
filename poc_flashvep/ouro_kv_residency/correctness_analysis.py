"""Compare initialized BF16 pages and native-layout outputs without GPU work."""
import json
import torch
from .common import ROOT,FIXTURES,write

def main():
    rows=[]
    for depth in range(4):
        for layer in (0,12,23):
            a=torch.load(FIXTURES/f'native_P2048_d{depth}_l{layer}.pt',weights_only=True)
            b=torch.load(FIXTURES/f'layer_major_P2048_d{depth}_l{layer}.pt',weights_only=True)
            item=dict(depth=depth,layer=layer,evidence='LIVE_MEASURED_OPERATOR',comparisons={})
            for key in ('q','k','v','out'):
                av,bv=a[key].float(),b[key].float()
                item['comparisons'][key]=dict(byte_exact=torch.equal(a[key],b[key]),
                    relative_L2=float(torch.linalg.vector_norm(av-bv)/torch.linalg.vector_norm(av).clamp_min(1e-12)),
                    absolute_max=float((av-bv).abs().max()))
            rows.append(item)
    a=json.loads((ROOT/'raw'/'correct_native'/'completed.json').read_text())['summary']['data']
    b=json.loads((ROOT/'raw'/'correct_layer_major'/'completed.json').read_text())['summary']['data']
    output=dict(requests=len(a['requests']),tokens_exact=[r['tokens'] for r in a['requests']]==[r['tokens'] for r in b['requests']],
        depths_exact=[r['depths'] for r in a['requests']]==[r['depths'] for r in b['requests']],
        output_cap=2,meaning='Bounded hardware correctness. Not natural benchmark accuracy.')
    write(ROOT/'layout_correctness.json',dict(rows=rows,outputs=output))
    print(json.dumps(dict(all_byte_exact=all(v['byte_exact'] for r in rows for v in r['comparisons'].values()),output=output)))

if __name__=='__main__':main()

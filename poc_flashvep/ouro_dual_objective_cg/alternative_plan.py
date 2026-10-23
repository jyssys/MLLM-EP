"""Register a small same-GPU fresh-state intervention on CALIBRATION sources."""
import gzip
import json
from .common import ROOT,GPUS,write

def main():
    assert not (ROOT/'p2_alternative_diagnostic_plan.json').exists()
    cases=json.loads(gzip.open(ROOT/'workload_payloads.json.gz','rt').read())['cases']
    pools={d:[r for r in cases if r['split']=='CALIBRATION' and r['domain']==d] for d in ('chat','long','stress8k')}
    jobs=[]
    for i,gpu in enumerate(GPUS):
        domain='long' if i%2==0 else 'stress8k'
        cells=[dict(label=f'FORCED_C{C}_G{G}',C=C,runtime='split' if G==1 else 'grouped',G=G)
            for G in (1,2) for C in (512,2048)]
        cells=cells[i:]+cells[:i]
        jobs.append(dict(id=f'dual_p2_ALTERNATIVE_{domain}_S{i}_gpu{gpu}_v1',kind='dual_serving',diagnostic_barrier=True,
            gpu=gpu,width=4,background_ids=[r['id'] for r in pools['chat'][i*3:i*3+3]],
            target_id=pools[domain][i]['id'],cells=cells,blocks=18432,budget=2048,
            graphs=True,streams=True,timeout_s=3600))
    write(ROOT/'p2_alternative_diagnostic_plan.json',dict(gpus=GPUS,jobs=jobs,
        scope='Same-GPU fixed C512/C2048 at G1/G2 after a synchronized background boundary. Verify state hashes before causal interpretation.'))
    print('Registered four controlled alternative-state jobs')

if __name__=='__main__':main()

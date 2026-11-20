"""Use original requests to examine length and B8192 chunk sensitivity."""
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .campaign import cell


def prepare():
    declaration=ROOT/'configs/frozen_input_bucket_control_v1.json'
    if declaration.exists():return json.loads(declaration.read_text())
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    ref=ROOT/'raw/BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1/natural_reference.json'
    gold={r['prompt_id']:r['output_tokens'] for r in json.loads(ref.read_text())['rows']}
    rng=np.random.default_rng(2026112086);buckets=[]
    for center,lo,hi in ((8192,6144,10240),(16384,14336,18432),(32768,28672,36864)):
        available=[i for i,r in enumerate(data) if lo<=r['input_tokens']<=hi]
        selected=sorted(map(int,rng.choice(available,min(8,len(available)),replace=False)))
        rows=[data[i] for i in selected];filename=f'input_bucket_{center}_original.json'
        write(ROOT/'data'/filename,rows)
        buckets.append(dict(center=center,range=[lo,hi],dataset=filename,N=len(rows),indices=selected,
            SHA256=digest(ROOT/'data'/filename),lengths=[r['input_tokens'] for r in rows],
            output_lengths=[gold[r['prompt_id']] for r in rows],prompt_ids=[r['prompt_id'] for r in rows]))
    result=dict(time=time.time(),seed=2026112086,QPS=2.35,arrival_seed=2026112087,cuts=[2,4,8,12],
        source_credit=4,actual_batch=[1,1,1],KV_block=128,B=8192,S=1024,buckets=buckets,
        scope='Original full requests in length buckets. No padding, cropping, output-based selection or hidden exact-length claim. '
        'Fixed per-request C1 output counts form a diagnostic, not the Natural EOS main benchmark. N<=8 and1start per cell do not establish a serving winner. '
        'The existing16-token prefix rule stays unchanged. Record actual aggregate GEMM-row mismatch. No exact-work claim if it is nonzero.')
    write(declaration,result);return result


def main():
    design=prepare()
    while not (ROOT/'state/credit_sweep_complete.json').exists() or not (ROOT/'state/runtime_api_profiles_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    budget=json.loads((ROOT/'state/budget.json').read_text());rows=[]
    for bucket in design['buckets']:
        for cut in (2,4,8,12):
            for axis in ('Layer','Token'):
                name=f'InputBucket{bucket["center"]}_{axis}{cut}_q2.35_c4_s2026112087_v1'
                if time.time()>budget['last_large_dispatch_epoch']-3600:
                    row=dict(name=name,status='NOT_RUN',reason='Reserve main confirmation and closeout budget.')
                else:
                    row=cell(name,axis,cut,dataset=bucket['dataset'],count=bucket['N'],seed=2026112087,
                        diagnostic_controlled_output_replay=True,replay_output_lengths=bucket['output_lengths'],
                        phase='ORIGINAL_LENGTH_BUCKET_CAUSAL_CONTROL')
                rows.append(dict(center=bucket['center'],**row))
                write(ROOT/'analysis/input_bucket_online_controls_v1.json',dict(results=rows,evidence='DIAGNOSTIC',
                    scope=design['scope']))
                assert row['status']!='FAILED',name
    write(ROOT/'state/input_bucket_controls_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':main()

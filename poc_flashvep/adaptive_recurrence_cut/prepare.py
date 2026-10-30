"""Freeze diagnostic prompt IDs and initial migration gate configurations."""
import argparse
import gzip
import json
from .common import ROOT,REPO,MODEL_PINS,CUTS,setup,write,digest


def main():
    setup()
    source=REPO/'poc_flashvep/reports/looped_pd_b_slo_serving/data/W1_long.json.gz'
    rows=json.loads(gzip.decompress(source.read_bytes()))
    if isinstance(rows,dict):rows=rows.get('requests',rows.get('data'))
    print('Frozen source',len(rows),list(rows[0]))
    data=[]
    for length in (128,512,1536):
        selected=[r for r in rows if all(len(r['input_ids'][m])>=length for m in MODEL_PINS if m in ('ouro','huginn'))][:8]
        if len(selected)<8:raise RuntimeError('Insufficient frozen source prompts')
        out=[]
        for index,r in enumerate(selected):
            out.append(dict(prompt_id=r.get('prompt_id',r.get('id',index)),
                            input_ids={m:r['input_ids'][m][:length] for m in ('ouro','huginn')},
                            source_tokens={m:len(r['input_ids'][m]) for m in ('ouro','huginn')},
                            source_sha256=digest(source),diagnostic_prefix_tokens=length))
        write(ROOT/'data'/f'prefix_{length}.json',out)
        data.append(dict(length=length,requests=len(out),source_sha256=digest(source)))
    write(ROOT/'configs/diagnostic_data_contract.json',dict(source=str(source),sha256=digest(source),
          origin='Preserved B PoC ShareGPT-derived token IDs. Prefix probes are not the new W1 serving workload.',datasets=data))
    for model in ('ouro','huginn'):
        depth=MODEL_PINS[model][2]
        for length in (128,512,1536):
            for cut in (depth,depth//2):
                name=f'gate_{model}_L{length}_cut{cut}'
                cfg=dict(name=name,model=model,cut=cut,gpus=[0,1] if model=='ouro' else [2,3],
                         count=8,cap=16,chunk=128 if length<=512 else 256,capacity=4,
                         state_capacity_tokens=4096,token_budget=128 if length<=512 else 256,
                         graphs=True,asynchronous=model=='ouro',warmup=True,seed=2026103051,
                         dataset=f'prefix_{length}.json',arrival_offsets=[i*.02 for i in range(8)],
                         verify_transfer=True,state_probe=True,ignore_eos=True,timeout=900,
                         prompt_ids=[r['prompt_id'] for r in json.loads((ROOT/'data'/f'prefix_{length}.json').read_text())])
                write(ROOT/'configs'/f'{name}.json',cfg)
    write(ROOT/'configs/stage_gates.json',dict(stage2_required_goodput_headroom_percent=10,
          stage2_requires_out_of_sample_simulator_validation=True,stage3_requires_stage2_GO=True,
          primary_joint_TTFT_TPOT_attainment=.9,co_batching=False,
          stage1_initial_scope='Diagnostic fixed cuts. No sustainable capacity claim.',
          source='Threshold frozen before any new GPU run.'))


if __name__=='__main__':main()

"""Compare saved native and TRUE EP observations without rerunning a model."""
import gzip
import json
import numpy as np
from .common import ROOT,write
from .aggregate_quality import csv_write
from .native_resume import directory as native_directory

def jsonlines(path):
    with path.open() as stream:return [json.loads(line) for line in stream]

def main():
    checks=[];generation=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        native={}
        with gzip.open(ROOT/'raw'/f'{model}_route_corpus_v2/router_corpus.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line)
                if row['split']!='heldout':continue
                key=(row['batch'],row['context'],row['request_id'],row['site'],row['loop'])
                assert key not in native
                native[key]=row
        for ep in [4,8]:
            directory=ROOT/'raw'/f'{model}_ep{ep}_full_profile_v1'
            for p in directory.glob('shape_fidelity.rank0.jsonl'):
                checks.extend(dict(row,model=model,ep=ep,type='NATIVE_LOGIT_AND_ALL_POSITION_TOP1',evidence='LIVE_SAVED_TENSORS') for row in jsonlines(p))
            p=directory/'routes.jsonl'
            routes={}
            if p.exists():
                for row in jsonlines(p):
                    if row['phase']!='prefill':continue
                    k=(row['batch'],row['context'])
                    value=routes.setdefault(k,dict(invocations=0,sampled_tokens=0,ids_exact=0,weights_exact=0,max_weight_abs=0.))
                    key=(row['batch'],row['context'],row['request_id'],row['site'],row['loop'])
                    ref=native[key]
                    assert row['positions']==ref['positions']
                    value['invocations']+=1;value['sampled_tokens']+=len(row['positions'])
                    actual=np.asarray(row['top2']);expected=np.asarray(ref['top2'])
                    value['ids_exact']+=int(np.all(actual==expected,axis=1).sum())
                    w=np.asarray(row['weights']);wr=np.asarray(ref['weights'])
                    value['weights_exact']+=int(np.all(w==wr,axis=1).sum())
                    value['max_weight_abs']=max(value['max_weight_abs'],float(np.abs(w-wr).max()))
                for (b,length),values in routes.items():checks.append(dict(model=model,ep=ep,batch=b,prefix=length,**values,
                    type='PROMPT_MATCHED_NATIVE_EP_ROUTE_SAMPLE',evidence='LIVE_AUTHORITATIVE_ROUTER_OUTPUT_SAMPLE',
                    all_sampled_routes_exact=values['sampled_tokens']==values['ids_exact']))
            for b in [1,2,4,8,16]:
                for length in [128,512,2048]:
                    ep_path=directory/f'generation_b{b}_p{length}.rank0.jsonl'
                    native_path=native_directory(model)/f'native_generation_b{b}_p{length}.rank0.jsonl'
                    row=dict(model=model,ep=ep,batch=b,prefix_tokens=length,eos_enabled=False,kv_status='N/A_FULL_PREFIX_NO_CACHE')
                    if ep_path.exists() and native_path.exists():
                        actual=jsonlines(ep_path)[0]['generated_ids'];expected=jsonlines(native_path)[0]['generated_ids']
                        eq=np.asarray(actual)==np.asarray(expected)
                        row.update(status='PASS' if eq.all() else 'DIFFERENT_OUTPUT',generated_tokens=eq.size,
                            equal_tokens=int(eq.sum()),exact_requests=int(eq.all(axis=0).sum()),requests=b,
                            first_divergent_step_1based=int(np.where(~eq.all(axis=1))[0][0]+1) if not eq.all() else '',evidence='LIVE_FREE_RUNNING_GREEDY_CAP32')
                    else:row.update(status='NOT_COMPLETE',evidence='NOT_MEASURED')
                    generation.append(row)
    path=ROOT/'correctness.jsonl'
    with path.open('w') as stream:
        for row in checks:stream.write(json.dumps(row)+'\n')
    csv_write(ROOT/'generation_fidelity.csv',generation)
    write(ROOT/'runs/fidelity_completeness.json',dict(check_rows=len(checks),completed_generation_cells=sum(v['status']!='NOT_COMPLETE' for v in generation),
        missing_cache_test_reason='Official paths use cache=None or use_cache=False. No retained KV state exists.'))

if __name__=='__main__':main()

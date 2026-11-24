"""Reproduce the saved C36 mismatch without launching or modifying a model."""
import csv
import json
import time

from .batch_prefix_probe import tensor_signature
from .common import ROOT,digest,preserve,write
from .numerical import embedding_metrics


def main():
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2);preserve()
    run='token_FROZEN_C36_F25_N8';base=run+'_0006'
    dpath=ROOT/'raw/batch_prefix'/f'{run}_D/manifest.json'
    ppath=ROOT/'raw/batch_prefix'/f'{run}_P/manifest.json'
    dm=json.loads(dpath.read_text());pm=json.loads(ppath.read_text())
    entry=dm['full_prompt_tensor_files'][base]
    assert digest(entry['path'])==entry['SHA256']
    actual=load_file(entry['path'])
    teacher_path=ROOT/'raw/llm_reference/token_FROZEN_C36_F25_N8_6_TEACHER_P/manifest.json'
    tm=json.loads(teacher_path.read_text());assert digest(tm['tensor_file'])==tm['tensor_SHA256']
    target=load_file(tm['tensor_file']);prefix=2221;length=2961
    expected_ids=tensor_signature(target['prompt_token_ids'][:prefix])
    expected_mrope=tensor_signature(target['full_prompt_mrope'][...,:prefix])
    prompt_match=pm['snapshots'][base]['prefix_ids']==expected_ids
    mrope_match=pm['snapshots'][base]['prefix_mrope']==expected_mrope
    layers=[];per_token=[]
    for layer in range(28):
        a,b=actual[f'prompt_KV_{layer}'],target[f'prompt_KV_{layer}']
        assert a.shape==b.shape and len(a)==length
        different=(a!=b).reshape(length,-1).any(dim=1)
        row=dict(layer=layer,strict_rows=int((~different).sum()),different_rows=int(different.sum()),
            first_different=different.nonzero().flatten().tolist()[:20])
        for label,left,right in [('initial_text',0,15),('prefix_visual',15,prefix),('suffix',prefix,length)]:
            row[label]=embedding_metrics(a[left:right],b[left:right])
        layers.append(row)
        af,bf=a.float().reshape(length,-1),b.float().reshape(length,-1)
        d2=(af-bf).square().sum(dim=1,dtype=torch.float64)
        b2=bf.square().sum(dim=1,dtype=torch.float64)
        relative=(d2/b2.clamp_min(1e-30)).sqrt().tolist()
        per_token += [dict(layer=layer,position=i,different=bool(different[i]),relative_L2=relative[i])
            for i in range(length)]
    chunks={}
    for role,record in [('P',pm),('D',dm)]:
        chunks[role]=[dict(batch=e['actual_batch'],graph=e['cudagraph_mode'],**r)
            for e in record['events'] for r in e['requests'] if r['base']==base and r['phase']=='prefill']
    teacher_chunks=[dict(start=e['start'],query_tokens=e['query_tokens']) for e in tm['events'] if e['start']<length]
    proof=dict(time=time.time(),evidence='TRACE_ANALYSIS and saved CORRECTNESS_ONLY state',
        passed=False,symptom='Frozen25% N8 has all28 prefix import hashes and all17 output tokens matching, but fails KV tolerance.',
        prompt_ids_match_teacher=prompt_match,mrope_match_teacher=mrope_match,chunks=chunks,
        teacher_chunks=teacher_chunks,layers=layers,actual_manifest_SHA256=digest(dpath),
        source_manifest_SHA256=digest(ppath),teacher_manifest_SHA256=digest(teacher_path),
        scope='Saved-state diagnosis only. No GPU replay, criterion relaxation or performance reclassification.')
    assert any(row['prefix_visual']['relative_L2']>.02 or row['suffix']['relative_L2']>.02 for row in layers)
    assert prompt_match and mrope_match
    write(ROOT/'analysis/frozen_C36_F25_Mantis6_saved_mismatch_v0.json',proof)
    with (ROOT/'analysis/frozen_C36_F25_Mantis6_row_errors_v0.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_token[0]));writer.writeheader();writer.writerows(per_token)
    print(json.dumps(dict(prompt_match=prompt_match,mrope_match=mrope_match,actual_chunks=chunks['P'],
        teacher_chunks=teacher_chunks,first_layer=layers[0]),indent=2));preserve()


if __name__=='__main__':main()

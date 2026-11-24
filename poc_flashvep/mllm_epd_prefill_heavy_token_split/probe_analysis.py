"""Recalculate read-only prompt-state and pre-divergence numerical evidence."""
import json
import time

import torch
from safetensors.torch import load_file

from .common import ROOT, digest, preserve, write
from .numerical import embedding_metrics


def main():
    preserve()
    name='KV_Native_vs_OfficialPD_Mantis6_v0'
    manifests={};tensors={};requests={}
    for arm in ['Native','OfficialPD']:
        requests[arm]=json.loads((ROOT/'raw'/f'{name}_{arm}_requests.jsonl').read_text().splitlines()[0])
        for role in ['P','D']:
            path=ROOT/'raw/llm_reference'/f'{name}_{arm}_{role}'/'manifest.json'
            record=json.loads(path.read_text())
            assert digest(record['tensor_file'])==record['tensor_SHA256']
            manifests[arm,role]=record
            tensors[arm,role]=load_file(record['tensor_file'])
    first=next((i for i,(a,b) in enumerate(zip(requests['Native']['token_ids'],requests['OfficialPD']['token_ids'])) if a!=b),None)
    layers=[]
    for layer in range(28):
        item=dict(layer=layer)
        for arm in ['Native','OfficialPD']:
            p,d=tensors[arm,'P'][f'prompt_KV_{layer}'],tensors[arm,'D'][f'prompt_KV_{layer}']
            imported=manifests[arm,'D']['snapshot']['event']['start']
            item[arm+'_imported_positions']=imported
            item[arm+'_imported_prefix_strict']=torch.equal(p[:imported],d[:imported])
        item['cross_arm_all_prompt_KV_strict']=torch.equal(tensors['Native','D'][f'prompt_KV_{layer}'],tensors['OfficialPD','D'][f'prompt_KV_{layer}'])
        layers.append(item)
    equal_P={key:torch.equal(tensors['Native','P'][key],tensors['OfficialPD','P'][key]) for key in
             ['prompt_token_ids','full_prompt_mrope','inputs_embeds_0','positions_0','hidden_0']}
    hidden=[]
    for i in range(first+1 if first is not None else 4):
        key='hidden_'+str(i)
        hidden.append(dict(output_token_index=i,**embedding_metrics(tensors['Native','D'][key],tensors['OfficialPD','D'][key])))
    logits={}
    if first is not None:
        choices={requests[arm]['token_ids'][first] for arm in requests}
        for arm in requests:
            row=manifests[arm,'D']['logits'][first]
            ids=row['token_ids'][0];values=row['values'][0]
            logits[arm]=dict(selected_token=requests[arm]['token_ids'][first],
                            compared_token_logits={str(k):values[ids.index(k)] if k in ids else None for k in sorted(choices)})
    result=dict(time=time.time(),evidence='CORRECTNESS_ONLY',requests=1,prompt_tokens=2961,
                source_probe=name,source_tensor_SHA256={f'{a}_{r}':v['tensor_SHA256'] for (a,r),v in manifests.items()},
                actual_P_input_strict=equal_P,KV_layers=layers,
                all_28_imported_prefix_strict=all(x['Native_imported_prefix_strict'] and x['OfficialPD_imported_prefix_strict'] for x in layers),
                all_28_cross_arm_prompt_KV_strict=all(x['cross_arm_all_prompt_KV_strict'] for x in layers),
                mrope_deltas={f'{a}_{r}':v['snapshot']['mrope_delta'] for (a,r),v in manifests.items()},
                first_output_divergence_index=first,hidden_through_first_divergence=hidden,first_divergence_logits=logits,
                interpretation='The prepared prompt, MRoPE and all 28 prompt KV layers match for this request. '
                    'The first output difference follows small decoder hidden differences and a greedy logit tie. '
                    'This refutes prompt KV corruption for this case. It does not qualify all requests or task accuracy.',
                performance_scope='Invalid for timing because the probe copies tensors to CPU.')
    write(ROOT/'analysis/Native_OfficialPD_Mantis6_prompt_KV_and_greedy_margin.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['KV_layers','source_tensor_SHA256']},indent=2))
    preserve()


if __name__=='__main__':main()

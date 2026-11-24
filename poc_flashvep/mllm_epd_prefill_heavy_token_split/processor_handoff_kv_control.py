"""Compare full prompt state for the same exact metadata control request."""
import asyncio
import json
import time

from .common import ROOT,write,preserve,digest
from .client import execute
from .llm_probe_control import rpc
from .source_snapshot import snapshot


def main():
    preserve();archive=snapshot('KV_Metadata_B2_Mantis6_v0')
    path=ROOT/'data/calibration_manifest_template_v2.json'
    row=json.loads(path.read_text())['requests'][6]
    name='KV_Metadata_B2_Mantis6_v0'
    write(ROOT/'configs'/f'{name}.json',dict(time=time.time(),request_id=row['request_id'],index=6,
        manifest_SHA256=digest(path),source_archive_SHA256=archive['archive_SHA256'],
        evidence='CORRECTNESS_ONLY',scope='Actual original model and full prompt KV. CPU copies invalidate all timing.'))
    for arm,url in [('None','http://127.0.0.1:18843'),('Grid','http://127.0.0.1:18845')]:
        active=[]
        try:
            for port,role in [(18841,'P'),(18842,'D')]:
                rpc(port,'llm_probe/begin',dict(name=name+'_'+arm+'_'+role,max_logits=16))
                active.append((port,role))
            result=asyncio.run(execute([row],[url],name+'_'+arm,.25,2026112402,None,True,
                metadata=dict(evidence='CORRECTNESS_ONLY',scope='No timing or throughput claim.')))
            assert result[0]['status']=='MEASURED_ONLINE'
        finally:
            for port,role in reversed(active):
                value=rpc(port,'llm_probe/end',{})
                write(ROOT/'state'/f'{name}_{arm}_{role}_close.json',value)
        print('KV_METADATA_CAPTURE_COMPLETE',arm,flush=True)
    import torch
    from safetensors.torch import load_file
    tensors={};manifests={};outputs={}
    for arm in ['None','Grid']:
        outputs[arm]=json.loads((ROOT/'raw'/f'{name}_{arm}_requests.jsonl').read_text().splitlines()[0])
        for role in ['P','D']:
            record=json.loads((ROOT/'raw/llm_reference'/f'{name}_{arm}_{role}/manifest.json').read_text())
            assert digest(record['tensor_file'])==record['tensor_SHA256']
            manifests[arm,role]=record;tensors[arm,role]=load_file(record['tensor_file'])
    fields={key:torch.equal(tensors['None','P'][key],tensors['Grid','P'][key]) for key in
        ['prompt_token_ids','full_prompt_mrope','inputs_embeds_0','positions_0','hidden_0']}
    layers=[]
    for layer in range(28):
        item=dict(layer=layer)
        for arm in ['None','Grid']:
            imported=manifests[arm,'D']['snapshot']['event']['start']
            item[arm+'_imported_positions']=imported
            item[arm+'_imported_prefix_strict']=torch.equal(tensors[arm,'P'][f'prompt_KV_{layer}'][:imported],
                tensors[arm,'D'][f'prompt_KV_{layer}'][:imported])
        item['cross_arm_prompt_KV_strict']=torch.equal(tensors['None','D'][f'prompt_KV_{layer}'],
            tensors['Grid','D'][f'prompt_KV_{layer}'])
        layers.append(item)
    proof=dict(time=time.time(),evidence='CORRECTNESS_ONLY',request_id=row['request_id'],
        P_prepared_fields_strict=fields,KV_layers=layers,
        all28_imported_prefix_strict=all(r['None_imported_prefix_strict'] and r['Grid_imported_prefix_strict'] for r in layers),
        all28_cross_arm_prompt_KV_strict=all(r['cross_arm_prompt_KV_strict'] for r in layers),
        mrope_deltas={f'{arm}_{role}':r['snapshot']['mrope_delta'] for (arm,role),r in manifests.items()},
        full_natural_token_match=outputs['None']['token_ids']==outputs['Grid']['token_ids'],
        tensor_SHA256={f'{arm}_{role}':r['tensor_SHA256'] for (arm,role),r in manifests.items()},
        scope='One same-geometry request on the same GPU group. This does not prove all cross-batch outputs or full task quality.')
    proof['passed']=all(fields.values()) and proof['all28_imported_prefix_strict'] and proof['all28_cross_arm_prompt_KV_strict'] and proof['full_natural_token_match']
    write(ROOT/'analysis'/f'{name}.json',proof)
    assert proof['passed'], 'Exact processor handoff changed the prompt state'
    preserve();print('KV_METADATA_ALL28_PASS',flush=True)


if __name__=='__main__':main()

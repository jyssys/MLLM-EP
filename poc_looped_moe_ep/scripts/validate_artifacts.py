"""Offline physical trace/ownership/provenance checks; no GPU allocation."""
import hashlib
import json
import math
from safetensors import safe_open
from huggingface_hub import snapshot_download
from poc_looped_moe_ep.contract import *


def main():
    snapshot=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    checks=[]
    module_prefix={'head':'model.head_layers.0','core':'model.stack.layers.0','tail':'model.tail_layers.0'}
    path=snapshot/SUBFOLDER/'model.safetensors'
    with safe_open(path,framework='pt',device='cpu') as weights:
        elements={}
        for name in weights.keys():
            tensor=weights.get_slice(name)
            elements[tensor.get_dtype()]=elements.get(tensor.get_dtype(),0)+math.prod(tensor.get_shape())
        for arm in ['S0','S1']:
            all_owners={pool:[] for pool in module_prefix}
            for rank in range(4):
                rows=json.loads((ROOT/f'results/raw/held_out_paired/ownership.{arm}.rank{rank}.json').read_text())
                for row in rows:
                    ids=row['owned_experts'];all_owners[row['pool']]+=ids
                    for projection in row['tensors']:
                        tensor=weights.get_slice(module_prefix[row['pool']]+'.'+projection['name'])
                        expected=[tensor[e,0,:8].float().tolist() for e in ids]
                        assert projection['samples']==expected
                        assert projection['dtype']=='torch.bfloat16'
                        assert projection['bytes']==len(ids)*1024*1536*2
                        checks.append(dict(arm=arm,rank=rank,pool=row['pool'],
                            projection=projection['name'],owned=ids,sample_exact=True))
            for pool,ids in all_owners.items():
                assert sorted(ids)==list(range(64 if pool=='core' else 8))
    source=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_moe_ep_poc_spec.md')
    assert sha(source)==sha(ROOT/'authoritative_spec.md')
    environment=json.loads((ROOT/'results/environment.json').read_text())
    assert environment['spec_sha256']==sha(source)
    write(ROOT/'results/model_manifest.json',dict(model=MODEL,revision=REVISION,subfolder=SUBFOLDER,
        actual_weight_file_bytes=path.stat().st_size,actual_weight_file_sha256=sha(path),
        actual_HF_modeling_code_sha256=sha(snapshot/'modeling_loop_lm.py'),
        actual_loop_trace_code_sha256=sha(snapshot/'loop_trace.py'),
        tensor_elements_by_dtype=elements,learned_parameters=elements['BF16'],
        unique_physical_sparse_pools=3,core_pool_experts=64,head_tail_experts=8,top_k=2,
        native_H=16,core_physical_layers=1,unrolled_attention_sites=18,
        per_pass_attention='16 untied modules; head/tail separate',
        router='one shared core router; separate head/tail routers',
        core_expert_sharing='one registered three-tensor pool called 16 times, not 16 copied expert modules',
        expert_bytes=9*1024**2,core_pool_bytes=576*1024**2,
        core_owned_per_EP4_rank=16,routed_bytes_per_EP4_rank=180*1024**2,
        native_cache=False,reference='official HF forward + local public/private grouped-mm API alias',
        precision='native BF16 learned tensors; exact original FP32 RoPE buffers restored',
        scope='small-model mechanism test; not representative four-H100 serving'))
    write(ROOT/'results/owner_weight_identity.json',dict(status='PASS',checks=checks,
        scope='8 exact values per projection per owned expert, all physical sparse pools, S0 and S1',
        full_tensor_hash_claim=False))
    print('Owner/checkpoint checks:',len(checks),'exact spec:',sha(source))


if __name__=='__main__':main()

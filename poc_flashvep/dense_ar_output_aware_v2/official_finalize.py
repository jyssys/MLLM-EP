"""Verify the independent compiled reference before releasing quality checks."""
import json
import time
from .common import ROOT,write,digest,dispatch_allowed

def main():
    contract=json.loads((ROOT/'configs/c3_quality_contract_v2.json').read_text());name=contract['primary_reference']
    folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
    while not receipt.exists():assert dispatch_allowed();time.sleep(5)
    result=json.loads(receipt.read_text());assert result['complete'],result
    cfg=json.loads((folder/'configuration.json').read_text())
    runtime=json.loads((folder/'actual_runtime_single.json').read_text())
    assert cfg['reference_single'] and cfg['serial'] and not cfg.get('axis_kind') and not cfg['ignore_eos']
    assert cfg['execution_family']=='INSTALLED_DEFAULT_COMPILE' and runtime['compile_mode']=='3'
    assert runtime['model_forward_module']=='vllm.model_executor.models.qwen3'
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text());rows=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
    assert len(rows)==len(data)==cfg['count'] and {r['prompt_id'] for r in rows}=={r['prompt_id'] for r in data}
    write(ROOT/'state/official_compiled_c1_reference_complete_v2.json',dict(time=time.time(),passed=True,name=name,n=len(rows),
        dataset_SHA256=cfg['dataset_SHA256'],source_capsule_SHA256=result['source_SHA256'],
        requests_SHA256=digest(folder/'requests.jsonl'),actual_runtime=runtime,
        cap_hits=sum(r.get('cap_hit',False) for r in rows),evidence='CORRECTNESS_ONLY',
        scope='Original official Full36 compiled single-GPU C1 reference. No partition adapter,imported KV or tensor correction. '
            'Quality reference only; its serial throughput is not continuous-serving capacity. Frozen Main400 IDs and Mode0 selection lengths are unchanged.'))
    print('OFFICIAL_COMPILED_REFERENCE_VERIFIED',len(rows),flush=True)

if __name__=='__main__':main()

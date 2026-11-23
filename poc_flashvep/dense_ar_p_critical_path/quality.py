"""Use the unchanged independent compiled C1 reference and quality margin."""
import json
import time
from .common import ROOT,PRIOR,write,digest


def compare(name,resamples=4000):
    from ..dense_ar_output_aware_v2.quality_gate import paired_quality
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    assert not cfg['ignore_eos']
    reference=json.loads((PRIOR/'state/official_compiled_c1_reference_complete_v2.json').read_text());assert reference['passed']
    source=PRIOR/'raw'/reference['name']/'requests.jsonl'
    refs={r['prompt_id']:r for r in map(json.loads,source.read_text().splitlines())}
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    actual=list(map(json.loads,(ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()))
    assert {r['prompt_id'] for r in actual}=={r['prompt_id'] for r in data}
    result=paired_quality(actual,[refs[r['prompt_id']] for r in data],data,resamples)
    contract=ROOT/'configs/c3_quality_contract_v2.json'
    margin=json.loads(contract.read_text())['paired_task_macro_margin_percentage_points']
    result.update(name=name,n=len(data),passed=result['CI95_percentage_points'][0]>=-margin,
        margin_percentage_points=margin,resamples=resamples,dataset_SHA256=cfg['dataset_SHA256'],
        contract_SHA256=digest(contract),reference=reference['name'],reference_requests_SHA256=digest(source),
        evidence='TASK_QUALITY_NONINFERIORITY',scope='Unchanged task macro metric and context-cluster bootstrap. '
            'The original compiled C1 reference stays frozen. This is not a server-start performance CI.')
    write(ROOT/'analysis'/(name+'_quality_gate.json'),dict(time=time.time(),**result))
    return {k:v for k,v in result.items() if k!='rows'}

"""Compare immutable diagnostic outputs and locate the first tensor difference."""
import argparse
from .common import *
from .analyze import records


def tensor_difference(x,y):
    import torch
    if x is None or y is None:return dict(equal=x is y)
    delta=(x.float()-y.float()).abs()
    return dict(equal=torch.equal(x,y),max_abs=float(delta.max()),mean_abs=float(delta.mean()),
        different_elements=int((x!=y).sum()),elements=x.numel())


def output_audit():
    reference=records(PREVIOUS/'raw/coarse_O16_N64_q4.777_native16_s1_v1/requests.jsonl')
    indexed={tuple(r['input_ids']):r for r in reference};result=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists():continue
        cfg=json.loads((ROOT/'configs'/(folder.name+'.json')).read_text())
        if cfg['cap']!=16:continue
        rows=records(folder/'requests.jsonl') if (folder/'requests.jsonl').exists() else []
        failures=[]
        for row in rows:
            expected=indexed[tuple(row['input_ids'])]
            if row['output_ids']==expected['output_ids']:continue
            first=next((i for i,(a,b) in enumerate(zip(row['output_ids'],expected['output_ids'])) if a!=b),-1)
            failures.append(dict(sample_id=row['sample_id'],prompt_id=row['prompt_id'],first_different_output_index=first))
        runtime=json.loads(receipt.read_text())
        result.append(dict(name=folder.name,runtime=runtime['status'],compared=len(rows),mismatches=failures,
            exact_output_passed=bool(rows) and len(rows)==cfg['count'] and not failures,
            scope='Original token reference. Matching prompt token IDs, not local sample numbering. This is not a KV/Hidden qualification.'))
    write(ROOT/'analysis/phase1_output_audit.json',dict(evidence='MEASURED',runs=result))
    return result


def continuous_state_gate():
    name='normfix_Continuous_budget8192_full_state_N32_v1';folder=ROOT/'raw'/name
    expected={r['external_request_id']:dict(KV=r['KV'],final_hidden_SHA256=r['final_hidden_SHA256'])
        for path in (PREVIOUS/'raw/qual_D16_independent_full_state_N32_v2').glob('extension_prefill_*.jsonl')
        for r in records(path) if r['event']=='VALID_QUERY_STATE' and r['external_request_id'].startswith('q')}
    actual={key:value for path in folder.glob('full_prompt_state_*.json')
        for key,value in json.loads(path.read_text())['requests'].items()}
    checks=[dict(request=key,all36_KV_bitwise_equal=actual.get(key,{}).get('KV')==value['KV'],
        all8000_hidden_bitwise_equal=actual.get(key,{}).get('final_hidden_SHA256')==value['final_hidden_SHA256'],
        queries=actual.get(key,{}).get('queries')) for key,value in expected.items()]
    lifetime={role:json.loads((folder/f'KV_lifetime_{role}.json').read_text()) for role in ('replica0','replica1')}
    output=next(row for row in output_audit() if row['name']==name)
    result=dict(name=name,independent_reference='previous/qual_D16_independent_full_state_N32_v2',
        compared=len(checks),checks=checks,lifetime=lifetime,output=output,
        passed=len(checks)==32 and all(r['all36_KV_bitwise_equal'] and r['all8000_hidden_bitwise_equal'] for r in checks)
            and all(v['passed'] for v in lifetime.values()) and output['exact_output_passed'],
        evidence='CORRECTNESS_ONLY',scope='Exclude warm-up IDs. Keep the original independent full Native hashes and token sequences. CPU state copies are not timing measurements.')
    write(ROOT/'analysis/continuous_N32_full_state_gate_v2.json',result)
    if result['passed']:write(ROOT/'state/continuous_N32_qualified.json',dict(time=time.time(),result=result))
    return result


def numeric(native,candidate,tag):
    import torch
    a=torch.load(ROOT/'raw'/native/'numeric_probe_decode.pt',weights_only=False)
    b=torch.load(ROOT/'raw'/candidate/'numeric_probe_replica1.pt',weights_only=False)
    result=dict(native=native,candidate=candidate,Native_batch=a['batch'],Continuous_batch=b['batch'],
        logits=tensor_difference(a['logits'],b['logits']),layers=[],submodules=[],
        evidence='CORRECTNESS_ONLY',scope='CPU tensor copies alter scheduling. These captures do not measure serving latency or physical idle.')
    for label,value in (('Native',a),('Continuous',b)):
        top=torch.topk(value['logits'],10)
        result[label+'_top10']=dict(tokens=top.indices.tolist(),logits=top.values.tolist())
    for index in range(36):
        name=f'model.layers.{index}.self_attn.attn'
        item=dict(layer=index,KV_prefix_bitwise_equal=a['KV'][index]['prefix_SHA256']==b['KV'][index]['prefix_SHA256'])
        for key in ('input_hidden','input_residual','output_hidden','output_residual'):
            item[key]=tensor_difference(a['layers'][index][key],b['layers'][index][key])
        for key in ('Q','K','V','output'):item['Attention_'+key]=tensor_difference(a['Attention'][name][key],b['Attention'][name][key])
        result['layers'].append(item)
    for name in a.get('submodules',{}):
        other=b.get('submodules',{}).get(name)
        if other is None:continue
        item=dict(name=name)
        for key in ('inputs','outputs'):
            item[key]=[tensor_difference(x,y) for x,y in zip(a['submodules'][name][key],other[key])]
        result['submodules'].append(item)
    write(ROOT/'analysis'/(tag+'.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--numeric',nargs=3);parser.add_argument('--state-gate',action='store_true');args=parser.parse_args()
    if args.state_gate:
        result=continuous_state_gate();print('CONTINUOUS_FULL_STATE_GATE',result['passed'],result['compared']);return
    if args.numeric:
        result=numeric(*args.numeric)
        print(json.dumps(result['submodules'],indent=2))
    else:
        for row in output_audit():print(row['name'],row['runtime'],row['compared'],len(row['mismatches']),row['exact_output_passed'])


if __name__=='__main__':main()

"""Verify prompt coverage,lifetimes,full KV hashes and numerical sample errors."""
import collections
import json
import math
import numpy as np
import torch
from .common import ROOT,write,digest


def intervals_cover(intervals,left,right):
    cursor=left
    for start,end in sorted(intervals):
        assert start==cursor and end>start,('Missing or duplicated query interval',intervals,left,right,cursor)
        cursor=end
    assert cursor==right,('Incomplete query coverage',intervals,left,right,cursor)


def audit(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    output=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    assert len(output)==len(data)
    result={};states={};decode_visits=collections.defaultdict(list)
    for role in ('prefill','decode'):
        assert json.loads((folder/f'KV_lifetime_{role}.json').read_text())['passed']
        hp=folder/f'hidden_lifetime_{role}.json'
        if hp.exists():assert json.loads(hp.read_text())['passed']
        events=json.loads((folder/f'axis_events_{role}.json').read_text());states[role]=events['states']
        parts={r['request_id']:r for r in events['rows'] if r['event']=='PARTITION'}
        visits=collections.defaultdict(list)
        for row in events['rows']:
            if row['event']!='MODEL_VISIT' or not row['request_id'].startswith('q'):continue
            length=parts[row['request_id']]['length']
            if role=='decode' and row['start']+row['count']>length:
                for layer in row['layers']:
                    decode_visits[row['request_id'],layer].append((max(length,row['start']),row['start']+row['count']))
            if role=='decode' and not parts[row['request_id']]['upper'] and not parts[row['request_id']]['suffix'] and row['start']==length-1 and row['count']==1:
                continue  # The official Native proxy recomputes one prompt query for its first logit.
            end=min(length,row['start']+row['count'])
            if row['start']>=end:continue
            for layer in row['layers']:visits[row['request_id'],layer].append((row['start'],end))
        for index,prompt in enumerate(data):
            rid='q'+str(index);part=parts[rid]
            for layer in range(36):
                if role=='prefill':left,right=(0,part['prefix']) if layer<part['lower'] else (0,0)
                else:left,right=(0,part['length']) if layer>=part['lower'] else (part['prefix'],part['length'])
                if left==right:assert not visits[rid,layer],(rid,role,layer,visits[rid,layer])
                else:intervals_cover(visits[rid,layer],left,right)
        result[role]=dict(prompt_query_layer_coverage=True,KV_lifetime=True,hidden_lifetime=True if hp.exists() else None)
    # The imported valid prefix KV must equal the producer's bytes exactly.
    imported=[]
    for index in range(len(data)):
        rid='q'+str(index)
        native_boundary=not parts[rid]['upper'] and not parts[rid]['suffix']
        source=states['prefill'].get('native_export_'+rid if native_boundary else rid)
        destination=states['decode'].get('import_'+rid)
        if source is None or destination is None:
            imported.append(dict(request_id=rid,status='NOT_CAPTURED'));continue
        assert source['length']==destination['length'] and source['layers']==destination['layers']
        matches=[a.get('digest_algorithm','RAW_SHA256')==b.get('digest_algorithm','RAW_SHA256') and
            a.get('digest',a.get('SHA256'))==b.get('digest',b.get('SHA256')) and a['bytes']==b['bytes']
            for a,b in zip(source['KV'],destination['KV'])]
        assert all(matches),(name,rid,'Imported KV differs from producer')
        imported.append(dict(request_id=rid,status='PASS',layers=len(matches),tokens=source['length']))
    queries=json.loads((folder/'actual_queries_decode.json').read_text())['rows'];groups=collections.defaultdict(list)
    for row in queries:groups[row['batch_id']].append(row)
    frozen_adapter=folder/'worker_source/poc_flashvep/dense_ar_longbench400_pd/adapter.py'
    full_graph_observer='def completed_full_graph' in frozen_adapter.read_text()
    decode_checks=[];emissions={r['sample_id']:r['output_tokens'] for r in output}
    for index,prompt in enumerate(data):
        rid='q'+str(index);length=prompt['input_tokens']
        expected=[(max(length,r['start']),r['start']+r['count']) for r in queries
            if r['request_id']==rid and r['start']+r['count']>length]
        end=max((right for _,right in expected),default=length)
        if expected:intervals_cover(expected,length,end)
        if not full_graph_observer:
            decode_checks.append(dict(request_id=rid,status='NOT_INSTRUMENTED',actual_decode_queries=end-length,
                emitted_output_tokens=emissions[index],full36_layer_coverage=None,
                scope='This frozen early source does not attach every captured-graph replay to a request. '
                    'Its prompt-state proof remains separate. Do not infer complete per-query decode observation.'))
            continue
        for layer in range(36):
            if end>length:intervals_cover(decode_visits[rid,layer],length,end)
            else:assert not decode_visits[rid,layer]
        assert end-length>=max(0,emissions[index]-1),(rid,'Fewer observed decode queries than emitted tokens')
        decode_checks.append(dict(request_id=rid,full36_layer_coverage=True,actual_decode_queries=end-length,
            emitted_output_tokens=emissions[index],extra_queries_over_R_minus_1=end-length-max(0,emissions[index]-1),
            scope='All executed post-prompt queries are counted. Any async lookahead work is not hidden.'))
    mixed=sum(any(r['API_prefilling'] for r in rows) and not all(r['API_prefilling'] for r in rows) for rows in groups.values())
    measured=dict(name=name,passed=True,schema_version=3,coverage=result,imported_prefix_KV=imported,
        decode_full36_checks=decode_checks,decode_full36_observation_complete=full_graph_observer,
        mixed_batches=mixed,
        maximum_actual_D_batch=max((r['requests'] for r in queries),default=0),n=len(data),
        evidence='CORRECTNESS_ONLY',
        scope='Coverage and valid imported-prefix hashes. Cross-axis model numerical equivalence requires the separate teacher comparison. '
            'Captured full36 decode graphs and actual query logs cover graph replays. No timing claim.')
    write(ROOT/'analysis'/(name+'_coverage.json'),measured);return measured


def tensor_error(actual,reference):
    a=actual.to(torch.float32);b=reference.to(torch.float32);error=a-b
    rms=float(torch.sqrt(torch.mean(error**2)));den=float(torch.sqrt(torch.mean(b**2)))
    return dict(max_absolute=float(error.abs().max()),RMS=rms,relative_RMS=rms/max(den,1e-30),
        cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)),
        exact=torch.equal(actual,reference))


def compare_single(teacher,arms):
    base=ROOT/'raw'/teacher
    gold=json.loads((base/'axis_events_prefill.json').read_text())['states']['q0']
    gold_tensor=torch.load(gold['tensor_file'],map_location='cpu',weights_only=False)
    tokens=json.loads((base/'requests.jsonl').read_text())['output_ids'];results=[]
    for name in arms:
        folder=ROOT/'raw'/name
        state=json.loads((folder/'axis_events_decode.json').read_text())['states']['q0']
        tensor=torch.load(state['tensor_file'],map_location='cpu',weights_only=False)
        positions=sorted(set(gold_tensor['positions'])&set(tensor['positions']))
        errors=[]
        for i,(actual,reference) in enumerate(zip(tensor['KV_samples'],gold_tensor['KV_samples'])):
            a=actual[[tensor['positions'].index(p) for p in positions]]
            b=reference[[gold_tensor['positions'].index(p) for p in positions]]
            errors.append(dict(layer=i,**tensor_error(a,b)))
        hidden=tensor_error(tensor['hidden'][-1],gold_tensor['hidden'][-1])
        match=json.loads((folder/'requests.jsonl').read_text())['output_ids']==tokens
        results.append(dict(name=name,strict_output_match=match,
            full_KV_exact_layers=sum(a['SHA256']==b['SHA256'] for a,b in zip(state['KV'],gold['KV'])),
            total_layers=36,KV_sample_errors=errors,last_hidden_error=hidden,shared_sample_positions=positions))
    result=dict(teacher=teacher,results=results,evidence='CORRECTNESS_ONLY',
        scope='Full valid KV hashes detect exact equality. Sample tensors quantify non-identical BI0 geometry. '
            'Sample errors are not a full-state numerical bound. Greedy output is tested independently.')
    write(ROOT/'analysis/N1_teacher_comparison_v1.json',result);return result


if __name__=='__main__':
    import sys
    print(json.dumps(audit(sys.argv[1]),indent=2))

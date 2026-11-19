"""Validate an actual generic cohort, without equating admission and batch."""
import collections
import hashlib
import json
import zipfile
from .common import ROOT,write
from .historical_audit import records


def validate_shapes(method,batch,source,destination,count,output):
    observed={}
    for role,values in (('prefill',source),('decode',destination)):
        counts=collections.Counter()
        for key,n in values.items():
            shape=json.loads(key)
            assert shape['query_tokens']<=8192
            if role=='prefill':
                assert shape['requests']==1 and shape['prefilling']
                label='source_prefill1'
            elif method=='Native':
                assert shape['requests']==shape['query_tokens']==shape['padded_query_tokens']==batch
                label='native_cached_full36_decode'+str(batch)
            elif shape['prefilling']:
                assert shape['requests']==1
                label='suffix_prefill1'
            else:
                assert shape['requests']==shape['query_tokens']==shape['padded_query_tokens']==batch
                label='full36_decode'+str(batch)
            counts[label]+=n
        observed[role]=dict(counts)
    assert observed['prefill']==dict(source_prefill1=count)
    expected={('native_cached_full36_decode' if method=='Native' else 'full36_decode')+str(batch):count*(output-1)//batch}
    if method!='Native':
        expected['suffix_prefill1']=count
    assert observed['decode']==expected,(observed,expected)
    return observed


def audit(name):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    batch=cfg['matched_D_cohort_size']
    assert cfg['P_group_size']==1 and cfg['D_admission_batch']==batch
    receipt=json.loads((folder/'dispatcher_result.json').read_text())
    assert receipt['complete'] and receipt['normal_worker_exit'] and receipt['timing_isolation_passed']
    assert json.loads((folder/'unified_worker_gate.json').read_text())['passed']
    hashes=json.loads((ROOT/'state/fixed_cohort2_v3_qualification_design.json').read_text())['runtime_adapter_hashes']
    with zipfile.ZipFile(folder/'source.zip') as archive:
        for filename,expected in hashes.items():
            assert hashlib.sha256(archive.read('poc_flashvep/dense_ar_matched_axis_pd/'+filename)).hexdigest()==expected
    shapes=[json.loads((folder/f'measured_geometry_{role}.json').read_text())['counts'] for role in ('prefill','decode')]
    observed=validate_shapes(cfg['evaluation_arm'],batch,*shapes,cfg['count'],cfg['cap'])
    groups=collections.defaultdict(list)
    for row in records(folder/'D_host_admission.jsonl'):
        groups[row['cohort']].append(row['request_id'])
    assert len(groups)==cfg['count']//batch and all(len(ids)==batch for ids in groups.values())
    barriers=records(folder/'matched_first_token_barrier.jsonl')
    assert len(barriers)==cfg['count']//batch
    assert sorted(r for row in barriers for r in row['members'])==sorted('q'+str(i) for i in range(cfg['count']))
    assert all(row['original_generation_timestamps_preserved'] for row in barriers)
    result=dict(name=name,passed=True,actual_P_prefill_batch=1,
        actual_D_prefill_batch=(0 if cfg['evaluation_arm']=='Native' else 1),
        D_prefill_batch_protocol=1,
        actual_D_decode_batch=batch,observed=observed,runtime_adapter_hashes=hashes,
        evidence='TRACE_ANALYSIS',scope='Fresh actual geometry and cohort-lifetime checks. The generic v3 adapter is unchanged. This audit does not replace a new-batch full-state qualification.')
    write(ROOT/'analysis'/(name+'_fixed_cohort_gate.json'),result)
    return result

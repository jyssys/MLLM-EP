"""Check real preparation1/Decode2 inputs and the first-token barrier."""
import argparse
import collections
import hashlib
import json
import time
import zipfile
from .common import ROOT, write
from .host_paths import records


def audit(name):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    assert cfg['matched_D_cohort_size']==cfg['D_admission_batch']==2
    assert cfg['P_group_size']==1
    receipt=json.loads((folder/'dispatcher_result.json').read_text())
    assert receipt['complete'] and receipt['normal_worker_exit'] and receipt['timing_isolation_passed']
    assert json.loads((folder/'unified_worker_gate.json').read_text())['passed']
    version=cfg.get('cohort_runtime_version',1)
    if version in (2,3):
        hashes=json.loads((ROOT/'state'/f'fixed_cohort2_v{version}_qualification_design.json').read_text())['runtime_adapter_hashes']
    else:
        freeze=json.loads((ROOT/'state/fixed_cohort2_runtime_freeze_before_performance.json').read_text())
        hashes={'fixed_matched_cohort.py':freeze['runtime_adapter_SHA256']}
    with zipfile.ZipFile(folder/'source.zip') as archive:
        for filename,expected in hashes.items():
            code=archive.read('poc_flashvep/dense_ar_matched_axis_pd/'+filename)
            assert hashlib.sha256(code).hexdigest()==expected
    observed={}
    for role in ('prefill','decode'):
        values=json.loads((folder/f'measured_geometry_{role}.json').read_text())['counts']
        counts=collections.Counter()
        for key, count in values.items():
            shape=json.loads(key)
            assert shape['query_tokens']<=8192
            if role=='prefill':
                assert shape['requests']==1 and shape['prefilling']
                label='source_prefill1'
            elif cfg['evaluation_arm']=='Native':
                # Native cached query1 is Decode work although the first
                # prepared-input flag says prefilling.
                assert shape['requests']==shape['query_tokens']==shape['padded_query_tokens']==2
                label='native_cached_full36_decode2'
            elif shape['prefilling']:
                assert shape['requests']==1
                label='suffix_prefill1'
            else:
                assert shape['requests']==shape['query_tokens']==shape['padded_query_tokens']==2
                label='full36_decode2'
            counts[label]+=count
        observed[role]=dict(counts)
    assert observed['prefill']==dict(source_prefill1=cfg['count'])
    if cfg['evaluation_arm']=='Native':
        assert observed['decode']==dict(native_cached_full36_decode2=cfg['count']*(cfg['cap']-1)//2)
    else:
        assert observed['decode']==dict(suffix_prefill1=cfg['count'],
            full36_decode2=cfg['count']*(cfg['cap']-1)//2)
    groups=collections.defaultdict(list)
    for row in records(folder/'D_host_admission.jsonl'):
        groups[row['cohort']].append(row['request_id'])
    assert len(groups)==cfg['count']//2 and all(len(v)==2 for v in groups.values())
    barriers=records(folder/'matched_first_token_barrier.jsonl')
    assert len(barriers)==cfg['count']//2
    assert sorted(r for b in barriers for r in b['members'])==sorted('q'+str(i) for i in range(cfg['count']))
    assert all(b['original_generation_timestamps_preserved'] and b['time']>=b['hold_begin'] for b in barriers)
    result=dict(name=name,passed=True,observed=observed,cohort_count=len(groups),
        original_generation_timestamps_preserved=True,
        first_token_hold_ms=[1000*(b['time']-b['hold_begin']) for b in barriers],
        runtime_adapter_hashes=hashes,runtime_version=version,
        evidence='TRACE_ANALYSIS',
        scope='Actual per-iteration inputs, not admission limits. Source prep1, suffix prep1 and steady Decode2 are equal across the two axes. Native has no suffix and keeps its cached initial full36 Decode2 query. The first-token barrier can increase TTFT. Worker first-notice-to-last spans can include remaining cohort preparation and are not exclusive Decode time.')
    write(ROOT/'analysis'/(name+'_fixed_cohort_gate.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',type=int,choices=(1,2,3),default=1)
    args=parser.parse_args()
    suffix=f'_v{args.version}' if args.version in (2,3) else ''
    design=json.loads((ROOT/'state'/('fixed_cohort2'+suffix+'_qualification_design.json')).read_text())
    results=[]
    for name in design['names']:
        path=ROOT/'raw'/name/'dispatcher_result.json'
        if path.exists() and json.loads(path.read_text()).get('complete'):
            results.append(audit(name))
    write(ROOT/'analysis'/('fixed_cohort2'+suffix+'_actual_geometry_audit.json'),dict(epoch=time.time(),
        starts=results,complete=len(results)==len(design['names']),evidence='TRACE_ANALYSIS'))
    print('FIXED_COHORT2_ACTUAL_GEOMETRY_AUDIT',len(results),len(design['names']),flush=True)


if __name__=='__main__':
    main()

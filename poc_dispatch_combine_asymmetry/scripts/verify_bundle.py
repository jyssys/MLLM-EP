"""Independent CPU-only recomputation from published compact raw evidence."""
import collections
import gzip
import hashlib
import json

import numpy as np

from poc_dispatch_combine_asymmetry.accounting import account
from poc_dispatch_combine_asymmetry.contract import ROOT, load, write
from poc_dispatch_combine_asymmetry.scripts.analyze import critical, ci


def main():
    index=load(ROOT/'artifacts/compact_raw/index.json')
    runs=collections.defaultdict(list)
    for row in index:
        content=gzip.decompress((ROOT/row['compact']).read_bytes())
        assert hashlib.sha256(content).hexdigest()==row['sha256_uncompressed'],row['compact']
        if '__stage.pid' in row['compact']:
            tag=row['compact'].split('/')[-1].split('__')[0]
            runs[tag].extend(map(json.loads,content.decode().splitlines()))
    final=load(ROOT/'artifacts/final_metrics.json')
    recomputed=collections.defaultdict(list)
    primary_waves=0
    for tag in ('mllm_clean_v2','llm_controls_v3'):
        waves=collections.defaultdict(list)
        for row in runs[tag]:
            if row['repeat']>=0 and row['lane']=='clean' and not row.get('overhead_pair'):
                waves[row['wave']].append(row)
        for rows in waves.values():
            value=critical(rows);row=rows[0];key=f"{row['model']}_{row['arm']}_{row['size']}_clean"
            recomputed[key].append(value);primary_waves+=1
    assert primary_waves==90 and len(recomputed)==9
    for key,rows in recomputed.items():
        assert len(rows)==10
        for metric in ('dispatch_ms','expert_ms','combine_ms','moe_ms','ratio'):
            assert ci([r[metric] for r in rows])==final['metrics'][key][metric],(key,metric)
    # The excluded attempt must still fail strict layer/rank validation.
    excluded=[r for r in runs['high_baseline_v1'] if r['repeat']==0]
    try:critical(excluded)
    except ValueError:pass
    else:raise AssertionError('known invalid first attempt was accepted')
    packed=np.load(ROOT/'artifacts/compact_routes.npz',allow_pickle=False)
    route_index=load(ROOT/'artifacts/compact_routes_index.json')
    groups=collections.defaultdict(dict)
    for row in route_index:groups[row['model'],row['arm'],row['size'],row['layer']][row['ep_rank']]=row
    accounting=load(ROOT/'artifacts/communication_accounting.json')['records']
    by_key={(r['model'],r['arm'],r['size'],r['layer']):r for r in accounting}
    assert len(groups)==len(by_key)==432
    for key,group in groups.items():
        assert set(group)==set(range(4))
        expected=by_key[key];n=expected['prompt_tokens'];width=(n+1)//2
        traffic=[]
        for rank,row in group.items():
            ids=packed[row['key']+'_ids'];owner=packed[row['key']+'_owner']
            assert owner.tolist()==[e-rank*32 if rank*32<=e<(rank+1)*32 else -1 for e in range(128)]
            weight=packed[row['key']+'_weights'];assert weight.shape==ids.shape and np.all(np.isfinite(weight))
            assert np.allclose(weight.sum(axis=1),1,atol=.01),key
            useful=min(width,max(0,n-rank*width)) if rank<2 else 0
            actual=account(ids[:useful],rank)
            assert actual==expected['logical'][rank],(key,rank)
            assert account(ids,rank)==expected['executed_or_padded'][rank]
            traffic.append(actual['dispatch_bytes_per_destination'])
        assert traffic==expected['dispatch_traffic_matrix']
        assert sum(sum(r) for r in traffic)==expected['logical_dispatch_bytes']==expected['logical_combine_bytes']
    profile=load(ROOT/'artifacts/nsight/kernel_classification.json')
    assert len(profile['summaries'])==12 and all(r['complete_48_layers'] for r in profile['summaries'])
    verdict=load(ROOT/'artifacts/final_verdict.json');assert not verdict['optimization_implemented']
    write(ROOT/'artifacts/bundle_verification.json',{'status':'PASS','GPU_REQUIRED':False,
        'raw_archives_sha256_verified':len(index),'primary_waves_independently_recomputed':primary_waves,
        'matched_clean_strata':9,'repeats_each':10,'compact_route_entries':len(route_index),
        'logical_accounting_layers_verified':432,'runtime_owner_maps_verified':432*4,
        'excluded_first_attempt_still_rejected':True,'profiler_native_ranges_complete':12,
        'scope':'exact CPU reaggregation/integrity, not another physical timing validation'})
    print('BUNDLE_VERIFY_PASS',primary_waves,'native waves;',len(groups),'route/accounting layers')


if __name__=='__main__':main()

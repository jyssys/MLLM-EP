"""Separate delivered latency, executed spans, and structural bytes."""
import csv,json,re,statistics
from .common import *
from .analyze import metrics,records,percentile
from .state_qualification import events


KEYS=('TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms','completed_QPS','SLO_goodput')


def work(name):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());folder=ROOT/'raw'/name
    requests=records(folder/'requests.jsonl');total=len(requests);cut=cfg.get('layer_cut',36)
    migration=cfg.get('layer_mode')=='MIGRATE'
    validated_span=not migration or cfg.get('separate_suffix_runner',False)
    source=events(name,'prefill');destination=events(name,'decode')
    mapping={x['request_id']:x['external_request_id'] for x in source+destination if x['event']=='REQUEST_IDENTITY'}
    is_measured=lambda x:bool(re.fullmatch(r'q\d+',mapping.get(x.get('request_id',''),'')))
    imports=[x for x in destination if x['event']=='HIDDEN_GPU_IMPORTED' and is_measured(x)]
    exports=[x for x in source if x['event']=='HIDDEN_GPU_EXPORTED' and is_measured(x)]
    closes={role:[x for x in events(name,role) if x['event']=='HIDDEN_GPU_POOL_CLOSE'] for role in ('prefill','decode')}
    native=[]
    for path in folder.glob('native_decode_*.json'):
        raw=json.loads(path.read_text())
        for x in raw['NIXL_stats_intervals']:native.append(x['data'])
    def stat(key):return [v for d in native for v in d.get(key,[])]
    counts=dict(complete_requests=total,migrated_requests=(total if migration else 0) if validated_span else 'INVALIDATED_DUPLICATE_PREFIX_EXECUTION',
        completed_D_prefill_layers=(total*(36-cut) if migration else 0) if validated_span else 'INVALIDATED_DUPLICATE_PREFIX_EXECUTION',
        P_prompt_layers=total*(cut if migration else 36),
        actual_hidden_export_count=len(exports),actual_hidden_import_count=len(imports),
        actual_hidden_import_bytes=sum(x['bytes'] for x in imports),
        Native_KV_bytes_transferred=sum(stat('bytes_transferred')),
        Native_KV_transfer_duration_p50_ms=percentile([1000*x for x in stat('transfer_duration')],.5),
        Native_KV_post_duration_p50_ms=percentile([1000*x for x in stat('post_duration')],.5),
        hidden_READ_host_p50_ms=percentile([x['transfer_request_host_ms'] for x in imports],.5),
        all_hidden_ACKed=all(len(v)==1 and v[0]['all_ACKed'] for v in closes.values()) if closes['prefill'] else 'NOT_MEASURED',
        layer_count_evidence='STRUCTURAL_ACCOUNTING_FROM_VERIFIED_STATIC_SPANS_AND_MEASURED_COMPLETED_REQUESTS' if validated_span else 'INVALIDATED_LEGACY_OPTIONAL_INPUT_DISPATCH',
        source='Native transfer telemetry is in seconds. Hidden READ host enclosures are not exclusive wire time. Layer totals are not independent per-layer kernel measurements.')
    per_layer=cfg['length']*8*128*2*2
    counts.update(structural_Native_KV_bytes_per_request=36*per_layer,
        structural_candidate_KV_plus_hidden_bytes_per_request=(cut*per_layer+cfg['length']*4096*2*2) if migration else 36*per_layer)
    return counts


def summarize(plan='serving_pilot_plan.json',output='serving_pilot_comparison.json'):
    names=json.loads((ROOT/'state'/plan).read_text())['jobs'];rows=[]
    for name in names:
        m=metrics(name)
        if m['complete']:rows.append(dict(name=name,metrics=m,work=work(name)))
    if not rows:return
    baseline=rows[0]['metrics']
    for row in rows:
        row['ratio_to_first_arm']={k:row['metrics'][k]/baseline[k] if baseline[k] else None for k in KEYS}
        row['Decode_noninferiority_point_pass']=row['ratio_to_first_arm']['ITL_p99_ms']<=1.05 and row['ratio_to_first_arm']['TPOT_p95_ms']<=1.05
    write(ROOT/'analysis'/output,dict(rows=rows,scope='Single pilot starts have no confirmation CI. Clean controls and event diagnostics remain separate.'))
    print(json.dumps([dict(name=r['name'],ratios=r['ratio_to_first_arm'],work=r['work']) for r in rows],indent=2))


if __name__=='__main__':
    import sys
    summarize(*sys.argv[1:])

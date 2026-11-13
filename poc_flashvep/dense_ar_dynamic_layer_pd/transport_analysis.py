"""Compare native transfer telemetry without claiming exclusive wire time."""
import collections
from .common import *
from .analyze import records,percentile,completed_D_layers


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    rows=records(folder/'requests.jsonl')
    if not rows:return None
    lifetime={};stats=collections.defaultdict(list);ack=[]
    for p in folder.glob('native_decode_*.json'):
        data=json.loads(p.read_text())
        for interval in data.get('NIXL_stats_intervals',[]):
            for k,v in interval['data'].items():stats[k].extend(v)
    hidden=[]
    for p in folder.glob('extension_decode_*.jsonl'):
        hidden.extend(r for r in records(p) if r['event']=='HIDDEN_GPU_IMPORTED' and r.get('request_id','').startswith('q'))
    expected_KV=sum((36-completed_D_layers(r,cfg))*cfg['length']*2*8*128*2 for r in rows) if cfg['arm']!='DP2' else 0
    expected_hidden=sum(completed_D_layers(r,cfg)>0 for r in rows)*cfg['length']*4096*2*2
    request_source={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    for p in folder.glob('native_prefill_*.json'):
        for r in json.loads(p.read_text()).get('NIXL_done_events',[]):
            for internal in r['request_ids']:
                rid=internal.split('-')[0]
                if rid in request_source:ack.append(1000*(r['time']-request_source[rid]))
    for role in ('prefill','decode'):
        path=folder/f'KV_lifetime_{role}.json'
        if path.exists():lifetime[role]=json.loads(path.read_text())
    def distribution(values):
        return dict(count=len(values),p50=percentile(values,.5),p95=percentile(values,.95),sum=sum(values))
    return dict(name=name,observed_prefix_KV_bytes=sum(stats['bytes_transferred']),
        observed_hidden_bytes=sum(r['bytes'] for r in hidden),structural_expected_prefix_KV_bytes=expected_KV,
        structural_expected_hidden_bytes=expected_hidden,KV_bytes_match=expected_KV==sum(stats['bytes_transferred']),
        hidden_bytes_match=expected_hidden==sum(r['bytes'] for r in hidden),
        transfer_duration_seconds=distribution(stats['transfer_duration']),post_duration_seconds=distribution(stats['post_duration']),
        descriptor_count=distribution(stats['num_descriptors']),
        hidden_submit_to_DONE_Host_ms=distribution([r['transfer_request_host_ms'] for r in hidden]),
        source_ready_to_native_done_Host_ms=distribution(ack),native_KV_lifetime=lifetime,
        failure_counts={k:sum(stats[k]) for k in ('num_failed_transfers','num_failed_notifications','num_kv_expired_reqs')},
        evidence='TRACE_ANALYSIS_AND_STRUCTURAL_ACCOUNTING',exclusive_wire_time='NOT_IDENTIFIED',
        scope='Native stats already convert NIXL microseconds to seconds. Transfer and post fields can overlap. Do not add or subtract them to infer exclusive CPU or wire time.')


def main():
    names=[]
    held=ROOT/'analysis/heldout_results.json'
    if held.exists():names=[r['name'] for r in json.loads(held.read_text())['results']]
    extra=ROOT/'analysis/exploratory_fraction50_results.json'
    if extra.exists():names+=[r['name'] for r in json.loads(extra.read_text())['results']]
    names+=['lean_native_O16_N64_v1','online_COST_O16_N64_v1']
    rows=[analyze(n) for n in names]
    write(ROOT/'analysis/transport_accounting.json',dict(results=[r for r in rows if r]))
    print('TRANSPORT_ACCOUNTING',len(rows))


if __name__=='__main__':main()

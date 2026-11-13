"""Report sampled clock associations without inferring causal latency shares."""
import collections,statistics
from .common import *
from .analyze import records,percentile


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    window=json.loads((folder/'window.json').read_text());requests=records(folder/'requests.jsonl')
    by_id={r['sample_id']:r for r in requests}
    offered=records(folder/'offered_requests.jsonl')
    placement={r['sample_id']:r.get('placement_role') for r in offered}
    intervals=collections.defaultdict(list)
    for index,r in by_id.items():
        gpu=cfg['gpus'][1]
        if cfg['arm']=='DP2':gpu=cfg['gpus'][0 if placement[index]=='replica0' else 1]
        intervals[gpu].append((r['first_token'],r['finish']))
    grouped=collections.defaultdict(list)
    for p in folder.glob('NVML-*.jsonl'):
        for r in records(p):
            if not window['start']<=r['time']<=window['end']:continue
            phase='HOST_DECODE_INTERVAL' if any(a<=r['time']<=b for a,b in intervals[r['physical_GPU']]) else 'OTHER_MEASUREMENT_INTERVAL'
            grouped[(r['physical_GPU'],phase)].append(r)
    result=[]
    for (gpu,phase),rows in grouped.items():
        values={}
        for field in ('SM_clock_MHz','memory_clock_MHz','power_W','temperature_C','sampled_utilization_percent','used_HBM_bytes'):
            series=[r[field] for r in rows if field in r]
            values[field]=dict(samples=len(series),mean=statistics.mean(series),minimum=min(series),
                p50=percentile(series,.5),p95=percentile(series,.95),maximum=max(series)) if series else 'NOT_MEASURED'
        masks=[r['clocks_event_reasons_mask'] for r in rows if 'clocks_event_reasons_mask' in r]
        values['clock_event_flags']=dict(samples=len(masks),raw_mask_counts=dict(collections.Counter(masks)),
            SW_power_cap_fraction=sum(bool(x&4) for x in masks)/len(masks) if masks else None,
            SW_or_HW_thermal_fraction=sum(bool(x&(32|64)) for x in masks)/len(masks) if masks else None)
        result.append(dict(GPU=gpu,phase=phase,samples=len(rows),observations=values))
    return dict(name=name,results=result,evidence='MEASURED_NVML_HOST_ALIGNED_ASSOCIATION',
        causal_DVFS_contribution='NOT_IDENTIFIED',
        source='https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html',
        scope='Decode phases use Host first-to-last token intervals. NVML samples do not locate individual kernels. Power averaging can span multiple phases. No clock or power setting is changed.')


def main():
    names=['frontier_Native_O16_D8_N64_v1','validation_fast_Static8_O16_N64_v1','online_COST_O16_N64_v1']
    held=ROOT/'analysis/heldout_results.json'
    if held.exists():names+=[r['name'] for r in json.loads(held.read_text())['results']]
    extra=ROOT/'analysis/exploratory_fraction50_results.json'
    if extra.exists():names+=[r['name'] for r in json.loads(extra.read_text())['results']]
    names+=[p.name for p in (ROOT/'raw').glob('enclosure_*_events_v1') if (p/'requests.jsonl').exists()]
    rows=[analyze(n) for n in names]
    write(ROOT/'analysis/clock_and_power_associations.json',dict(results=rows,
        scope='Clock associations can motivate a controlled follow-up. They do not identify an exclusive contribution to ITL or TPOT.'))
    print('CLOCK_ASSOCIATIONS',len(rows))


if __name__=='__main__':main()

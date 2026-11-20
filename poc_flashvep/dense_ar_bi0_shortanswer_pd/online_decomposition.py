"""Partition observed TTFT differences. Do not infer exclusive GPU causes."""
import collections
import json
import numpy as np
from .common import ROOT,write
from .axis_timeline import analyze

PARTS=('P_queue_ms','P_prepare_to_ready_ms','ready_to_D_prepare_ms',
       'D_prepare_to_first_worker_output_ms','first_worker_to_client_ms')


def compare(left,right):
    a=analyze(left);b=analyze(right)
    assert [r['prompt_id'] for r in a['rows']]==[r['prompt_id'] for r in b['rows']]
    delta={key:float(np.mean([r[key] for r in a['rows']])-np.mean([r[key] for r in b['rows']]))
           for key in PARTS+('TTFT_ms',)}
    assert abs(sum(delta[key] for key in PARTS)-delta['TTFT_ms'])<.001
    return dict(numerator=left,denominator=right,N=len(a['rows']),
        means={label:{key:float(np.mean([r[key] for r in value['rows']]))
                       for key in PARTS+('TTFT_ms','output_tokens')}
               for label,value in (('numerator',a),('denominator',b))},
        numerator_minus_denominator_ms={key:value for key,value in delta.items()},
        output_work_equal=all(x['output_tokens']==y['output_tokens'] for x,y in zip(a['rows'],b['rows'])))


def main():
    pairs=[]
    discovery=ROOT/'analysis/BI0_controlled_fixed_axis_discovery_v3.json'
    if discovery.exists():
        lookup={(r['axis'],r['cut']):r['name'] for r in json.loads(discovery.read_text())['results']}
        for cut in (1,2,4,6,8,12):
            if ('Layer',cut) in lookup and ('Token',cut) in lookup:
                pairs.append(dict(phase='DISCOVERY_ONE_START',cut=cut,
                    comparison=compare(lookup['Layer',cut],lookup['Token',cut])))
    for file in ('BI0_shared_KV128_confirmation_v2.json','BI0_static_QPS_extension_v1.json','BI0_controlled_low_QPS_followup_v1.json'):
        path=ROOT/'analysis'/file
        if not path.exists():continue
        groups=collections.defaultdict(dict)
        for row in json.loads(path.read_text())['results']:
            seed=row.get('seed')
            if seed is None:seed=json.loads((ROOT/'raw'/row['name']/'configuration.json').read_text())['arrival_seed']
            groups[(seed,row['metrics']['offered_QPS'])][row['label']]=row['name']
        for (seed,rate),group in groups.items():
            for left,right in (('Layer','Token'),('Layer','ControlledNative'),('Token','ControlledNative')):
                if left in group and right in group:
                    pairs.append(dict(phase=file,seed=seed,QPS=rate,
                        comparison=compare(group[left],group[right])))
    root=ROOT/'analysis/BI0_cut12_root_confirmation_v1.json'
    if root.exists():
        groups=collections.defaultdict(dict)
        for row in json.loads(root.read_text())['results']:groups[row['seed']][row['axis']]=row['name']
        for seed,group in groups.items():
            if 'Layer' in group and 'Token' in group:
                pairs.append(dict(phase=root.name,seed=seed,QPS=2.35,cut=12,
                    comparison=compare(group['Layer'],group['Token'])))
    value=dict(comparisons=pairs,evidence='TRACE_ANALYSIS',
        scope='Exact additive partition of observed mean client TTFT, not a decomposition of TTFT p95. '
        'Same prompt IDs are paired. Natural EOS output work can differ. P queue contains source submission delay. '
        'Ready-to-D-prepare includes transport, scheduling and admission. These intervals are not exclusive GPU kernel times. '
        'An additive difference locates the observed path difference. It does not establish a causal operator attribution.')
    write(ROOT/'analysis/BI0_online_TTFT_observed_decomposition_v1.json',value)
    return value


if __name__=='__main__':main()

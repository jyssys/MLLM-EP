"""Account for direct-D requests without inventing a producer or handoff."""
import json
import numpy as np
from .common import ROOT, write


def analyze(name):
    from .service_accounting import records
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    assert cfg.get('whole_prefill_deflection')
    assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
    requests={f'q{r["sample_id"]}':r for r in records(folder/'requests.jsonl')}
    direct={r['request_id'] for r in records(folder/'whole_prefill_deflections.jsonl')}
    assert direct<=set(requests)
    submissions={(r['request_id'],r['role']):r['time'] for r in records(folder/'request_submissions.jsonl')}
    admissions={(r['request_id'],r['role']):r['worker_accepted'] for r in records(folder/'admissions.jsonl')}
    ready={r['request_id']:r['source_ready'] for r in records(folder/'handoffs.jsonl')}
    queries={}
    for role in ('prefill','decode'):
        for r in json.loads((folder/f'actual_queries_{role}.json').read_text())['rows']:
            queries.setdefault((r['request_id'],role),r['time'])
    rows=[]
    for rid,r in sorted(requests.items(),key=lambda item:int(item[0][1:])):
        first=r['token_events'][0]['worker_time']
        d=[submissions[rid,'decode'],admissions[rid,'decode'],queries[rid,'decode'],first,r['first_token']]
        if rid in direct:
            assert (rid,'prefill') not in submissions and (rid,'prefill') not in admissions
            assert (rid,'prefill') not in queries and rid not in ready
            times=[r['arrival']]+d
            labels=['Arrival_to_D_submit_ms','D_submit_to_accept_ms','D_accept_to_prepare_ms',
                    'D_prepare_to_first_worker_ms','Worker_to_client_ms']
            route='D_LOCAL_FULL36'
        else:
            times=[r['arrival'],submissions[rid,'prefill'],admissions[rid,'prefill'],
                   queries[rid,'prefill'],ready[rid]]+d
            labels=['Arrival_to_P_submit_ms','P_submit_to_accept_ms','P_accept_to_prepare_ms',
                    'P_prepare_to_ready_ms','Ready_to_D_submit_ms','D_submit_to_accept_ms',
                    'D_accept_to_prepare_ms','D_prepare_to_first_worker_ms','Worker_to_client_ms']
            route='P_FULL36_THEN_D'
        stages={k:1000*(b-a) for k,a,b in zip(labels,times,times[1:])}
        assert abs(sum(stages.values())-r['TTFT_ms'])<1e-3
        rows.append(dict(request_id=rid,prompt_id=r['prompt_id'],route=route,
            input_tokens=r['input_tokens'],output_tokens=r['output_tokens'],TTFT_ms=r['TTFT_ms'],
            D_decode_residency_ms=1000*(r['finish']-r['first_token']),stages_ms=stages))
    groups={}
    for route in ('P_FULL36_THEN_D','D_LOCAL_FULL36'):
        subset=[r for r in rows if r['route']==route]
        groups[route]=dict(n=len(subset),means={k:float(np.mean([r[k] for r in subset]))
            for k in ('TTFT_ms','D_decode_residency_ms')} if subset else {},
            stage_means_ms={k:float(np.mean([r['stages_ms'][k] for r in subset]))
                            for k in subset[0]['stages_ms']} if subset else {})
    means={k:float(np.mean([r[k] for r in rows])) for k in ('TTFT_ms','D_decode_residency_ms')}
    result=dict(name=name,rows=rows,means=means,route_counts={k:v['n'] for k,v in groups.items()},
        route_groups=groups,evidence='TRACE_ANALYSIS',execution_scope='C3_CONTINUOUS_ROUTE_AWARE',
        scope='Direct-D requests have no P submission,compute or handoff. Those stages are not applicable,not fabricated zeros. '
              'Each route has its own additive observed mean-TTFT path. Conditional route means use only that route. '
              'Arrival-to-D-submit includes prior coordinator waiting and idle certification,not exclusive D queue time. '
              'Prepare is a host boundary. No GPU-exclusive service or TTFTp95 decomposition. '
              'Route choice is endogenous to current workload and D state,not randomized.')
    assert abs(sum(g['n']*g['means']['TTFT_ms'] for g in groups.values() if g['n'])/len(rows)-means['TTFT_ms'])<1e-3
    write(ROOT/'analysis'/(name+'_service_accounting.json'),result)
    return result


def paired(name,reference):
    from .service_accounting import analyze as generic
    a=analyze(name);b=generic(reference)
    configs=[json.loads((ROOT/'raw'/n/'configuration.json').read_text()) for n in (name,reference)]
    for field in ('source_capsule_SHA256','dataset_SHA256','arrival_offsets','seed','gpus','output_cap','ignore_eos'):
        assert configs[0][field]==configs[1][field],field
    right={r['request_id']:r for r in b['rows']}
    assert set(right)=={r['request_id'] for r in a['rows']}
    assert all(r['prompt_id']==right[r['request_id']]['prompt_id'] for r in a['rows'])
    groups={}
    for route,group in a['route_groups'].items():
        subset=[r for r in a['rows'] if r['route']==route]
        groups[route]=dict(group,reference_subset_mean_TTFT_ms=float(np.mean([
            right[r['request_id']]['TTFT_ms'] for r in subset])) if subset else None,
            paired_mean_TTFT_delta_ms=float(np.mean([
                r['TTFT_ms']-right[r['request_id']]['TTFT_ms'] for r in subset])) if subset else None)
    return dict(left=name,right=reference,seed=configs[0]['seed'],GPU_pair=configs[0]['gpus'],
        mean_TTFT_delta_ms=a['means']['TTFT_ms']-b['means']['TTFT_ms'],route_groups=groups,
        scope='Same-ID reference subsets. Route is selected by the deflection policy,not randomized. '
              'Do not use a direct-D path as a six-stage P-to-D path or assign its gain to P compute.')

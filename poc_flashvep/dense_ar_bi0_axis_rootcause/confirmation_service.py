"""Summarize measured host demand separately from GPU-exclusive service."""
import itertools
import json
import numpy as np
from .common import ROOT,write
from .causal_replay import extract
from .source_chain import analyze as source_chain
from .destination_chain import current_rows,decompose


def mean_interval(values):
    assert len(values)==3
    values=np.asarray(values)
    draws=[float(values[list(indices)].mean()) for indices in itertools.product(range(3),repeat=3)]
    return dict(mean=float(values.mean()),CI95=[float(np.percentile(draws,p)) for p in (2.5,97.5)],starts=3,resamples=27)


def main():
    stats=json.loads((ROOT/'analysis/full200_confirmation_statistics.json').read_text());rows=[]
    for cell in stats['comparisons']:
        for axis in ('Native','Layer','Token'):
            starts=[]
            for start in cell['absolute_start_means'][axis]['per_start']:
                name=start['name'];data=extract(name)['rows']
                timeline=json.loads((ROOT/'analysis'/(name+'_axis_timeline.json')).read_text())['summary']
                source=source_chain(name,ROOT)
                destination=decompose(current_rows(name))
                starts.append(dict(name=name,seed=start['seed'],
                    P_host_service_ms=1000*float(np.mean([r['P_service'] for r in data])),
                    D_first_host_service_ms=1000*float(np.mean([r['D_first_service'] for r in data])),
                    D_whole_host_demand_ms=1000*float(np.mean([r['D_first_service']+r['D_decode'] for r in data])),
                    P_queue_ms=timeline['P_queue_ms']['mean'],
                    ready_to_D_prepare_ms=timeline['ready_to_D_prepare_ms']['mean'],
                    source_chain=source,destination_chain=destination))
            fields=('P_host_service_ms','D_first_host_service_ms','D_whole_host_demand_ms','P_queue_ms','ready_to_D_prepare_ms')
            rows.append(dict(cut=cell['cut'],axis=axis,starts=starts,
                summary={field:mean_interval([r[field] for r in starts]) for field in fields}))
    differences=[]
    for cut in sorted({r['cut'] for r in rows}):
        pair={r['axis']:r for r in rows if r['cut']==cut and r['axis'] in ('Layer','Token')}
        source_parts={key:float(np.mean([r['source_chain']['exact_parts_ms'][key] for r in pair['Layer']['starts']])-
            np.mean([r['source_chain']['exact_parts_ms'][key] for r in pair['Token']['starts']]))
            for key in pair['Layer']['starts'][0]['source_chain']['exact_parts_ms']}
        observed=pair['Layer']['summary']['P_queue_ms']['mean']-pair['Token']['summary']['P_queue_ms']['mean']
        assert abs(sum(source_parts.values())-observed)<1e-3
        differences.append(dict(cut=cut,Layer_minus_Token_mean_P_queue_ms=observed,
            exact_weighted_source_chain_delta_ms=source_parts,
            rho_PD_descriptive={axis:r['summary']['P_host_service_ms']['mean']/r['summary']['D_whole_host_demand_ms']['mean'] for axis,r in pair.items()},
            mean_D_ready_supply_missing_fraction={axis:float(np.mean([s['destination_chain']['ready_supply_missing_fraction'] for s in r['starts']])) for axis,r in pair.items()}))
    write(ROOT/'analysis/full200_confirmation_host_service_demand.json',dict(rows=rows,
        source_chain_differences=differences,evidence='TRACE_ANALYSIS',
        scope='Natural200,QPS2.35,C0 actual1/1/1,c4,h4,KV128. Means of request means across three independent starts. '
        'Host demand includes launch,transport,scheduler and delivery costs. D-whole includes natural decode residency. '
        'These are neither GPU-exclusive times nor a validated single-knob decision boundary. Native starts are reused across cuts,not repeated measurements.'))
    print(json.dumps([dict(cut=r['cut'],axis=r['axis'],**{key:value['mean'] for key,value in r['summary'].items()}) for r in rows],indent=2))


if __name__=='__main__':main()

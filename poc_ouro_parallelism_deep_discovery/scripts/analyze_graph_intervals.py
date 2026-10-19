"""Keep cumulative graph envelopes separate from parent E2E wall time."""
import argparse
import collections
import csv
import statistics
from poc_ouro_parallelism_deep_discovery.contract import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run
    assert read(run/'status.json')['success'] and read(run/'guard_status.json')['success']
    target=run/'interval_analysis';target.mkdir(exist_ok=False)
    metadata=read(run/'observation.json');rows=[]
    for path in sorted(run.glob('context*_repeat*_intervals.json')):
        value=read(path);candidates=[]
        for rank in value['ranks']:
            events=rank['records']
            assert all(r['stream_before']==r['stream_after'] for r in events)
            graph=sorted((r for r in events if r['kind']=='graph_replay'),key=lambda r:r['start_ms'])
            assert len(graph)==127,(path,len(graph))
            assert all(b['start_ms']>=a['end_ms']-1e-4 for a,b in zip(graph,graph[1:]))
            total=sum(r['duration_ms'] for r in graph)
            span=graph[-1]['end_ms']-graph[0]['start_ms']
            candidates.append(dict(rank=rank['rank'],graph_sum_ms=total,graph_window_ms=span,
                unattributed_between_graph_ms=span-total,graph_fraction_of_local_window=total/span,
                graph_calls=len(graph),first_graph_ms=graph[0]['duration_ms'],
                median_graph_ms=statistics.median(r['duration_ms'] for r in graph),
                outer_model_calls=sum(r['kind']=='execute_model_outer' for r in events),
                sample_calls=sum(r['kind']=='sample_tokens_outer' for r in events)))
        winner=max(candidates,key=lambda r:r['graph_sum_ms'])
        rows.append(dict(context=value['context'],repeat=value['repeat'],
            selected_rank=winner['rank'],**{k:v for k,v in winner.items() if k!='rank'},
            parent_E2E_ms=value['request']['wall_ms'],
            parent_decode_span_ms=(value['request']['metrics']['last_token_ts']-value['request']['metrics']['first_token_ts'])*1000,
            selected_rank_rule='Maximum cumulative complete-graph duration. One rank retained for every displayed component.',
            scope='Selected-rank graph-window accounting. Not a full E2E component breakdown or removable-cost estimate.'))
        write(target/(path.stem+'_rank_summaries.json'),dict(ranks=candidates,selected_rank=winner['rank']))
    with (target/'request_rows.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=[]
    for context in sorted({r['context'] for r in rows}):
        selected=[r for r in rows if r['context']==context]
        control=next(r for r in metadata['records'] if r['context']==context)
        summaries.append(dict(context=context,N_trials=len(selected),N_distinct_request_shapes=1,
            median_graph_sum_ms=statistics.median(r['graph_sum_ms'] for r in selected),
            median_local_graph_window_ms=statistics.median(r['graph_window_ms'] for r in selected),
            median_unattributed_between_graph_ms=statistics.median(r['unattributed_between_graph_ms'] for r in selected),
            median_graph_fraction_of_local_window=statistics.median(r['graph_fraction_of_local_window'] for r in selected),
            median_graph_ms=statistics.median(r['median_graph_ms'] for r in selected),
            median_parent_E2E_ms=statistics.median(r['parent_E2E_ms'] for r in selected),
            observer_vs_before_fraction=control['observer_vs_before_fraction'],
            observer_vs_after_fraction=control['observer_vs_after_fraction'],
            before_after_fraction=control['before_after_fraction']))
    write(target/'summary.json',dict(size=metadata['size'],TP=metadata['TP'],physical_gpus=metadata['physical_gpus'],
        summaries=summaries,evidence_class='GPU_MICROBENCH',
        clock='Each rank retains its own event clock. No cross-rank timestamp alignment.',
        waiting='Graph envelopes can contain internal compute, communication, and waiting. Those components remain unresolved.',
        exclusions='Prefill, the first sampling interval, and the final sampling interval are outside the defined graph window.',
        headroom='The complement is unattributed, not a demonstrated removable cost. Complete graph contents could still contain other opportunities.',
        sample_unit='Three repeats of one frozen shape at each context. No broad request-level CI or candidate performance claim.'))
    for r in summaries:print(metadata['size'],r['context'],r['median_graph_ms'],r['median_graph_fraction_of_local_window'],r['observer_vs_before_fraction'],r['observer_vs_after_fraction'])

if __name__=='__main__':main()

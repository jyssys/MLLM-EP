"""Audit existing measured context costs before proposing a new input sweep."""
from .common import *
from .analyze import records,percentile


def main():
    name='length_cost_B2_all_cuts_contexts_N120_v1';folder=ROOT/'raw'/name
    rows={f"q{r['sample_id']}":r for r in records(folder/'requests.jsonl')}
    sends={r['request_id']:r['actual_send_start'] for r in records(folder/'actual_submissions.jsonl')}
    ready={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    events=json.loads((folder/'CUDA_event_enclosures_decode.json').read_text())['rows'];samples=[]
    for group in records(folder/'actual_cut_commits.jsonl'):
        ids=group['requests'];layers=group['offloaded_layers'];length=rows[ids[0]]['input_tokens']
        assert len(ids)==2 and all(rows[r]['input_tokens']==length for r in ids)
        first=[next(v for v in events if rid in v['requests']) for rid in ids]
        assert all(len(v['requests'])==2 for v in first)
        source=max(ready[r] for r in ids);start=min(v['start_monotonic'] for v in first)
        end=max(v['end_monotonic'] for v in first);output=max(rows[r]['first_token'] for r in ids)
        samples.append(dict(length=length,layers=layers,
            P_prefix_host_ms=1000*(source-min(sends[r] for r in ids)),
            handoff_host_ms=1000*(start-source),D_suffix_event_ms=max(v['elapsed_ms'] for v in first) if layers else 0.,
            commit_host_ms=max(0.,1000*(output-end)) if layers else 0.,
            first_ITL_host_ms=max(rows[r]['tpot_ms'] for r in ids)))
    summaries=[]
    for length in sorted({r['length'] for r in samples}):
        options=[]
        for layers in (0,2,4,8,12):
            values=[v for v in samples if v['length']==length and v['layers']==layers]
            assert len(values)==3
            value={k:percentile([v[k] for v in values],.95) for k in
                ('P_prefix_host_ms','handoff_host_ms','D_suffix_event_ms','commit_host_ms','first_ITL_host_ms')}
            # This is a modeled stationary service comparison, not serving.
            D=value['handoff_host_ms']+value['D_suffix_event_ms']+value['commit_host_ms']+15*value['first_ITL_host_ms']
            options.append(dict(layers=layers,samples=3,**value,
                modeled_O16_P_group_service_ms=value['P_prefix_host_ms'],
                modeled_O16_D_group_service_ms=D,
                modeled_stationary_bottleneck_ms=max(value['P_prefix_host_ms'],D)))
        best=min(options,key=lambda v:v['modeled_stationary_bottleneck_ms'])
        summaries.append(dict(length=length,modeled_best_layers=best['layers'],options=options))
    write(ROOT/'analysis/length_axis_measured_cost_and_modeled_balance.json',dict(name=name,
        raw_samples=samples,contexts=summaries,evidence='STRUCTURAL_ACCOUNTING',
        component_evidence='MEASURED',serving_benefit='NOT_MEASURED',
        scope='Existing exact Batch2 isolated calibration, three groups per context and cut. Source and handoff are Host intervals. Suffix is a CUDA Event enclosure, not kernel union. O16 repeats the measured cap2 first ITL15 times and ignores unknown queues, transport overlap and changing decode batch. It is a hypothesis screen, not a safe budget, capacity oracle or online result. The transport and synchronous Decode family differ from later async discovery.'))
    print('LENGTH_AXIS_MODEL',[(v['length'],v['modeled_best_layers']) for v in summaries],flush=True)


if __name__=='__main__':main()

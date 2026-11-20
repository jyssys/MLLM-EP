"""Measure direct credit blocking and its propagated source-chain delay."""
import json
from .common import ROOT,write
from .queue_accounting import audit,lines


def measure(name):
    folder=ROOT/'raw'/name;result=audit(folder)
    events=lines(folder/'credit_events.jsonl')
    acquired=[r for r in events if r['event']=='ACQUIRE_P_DISPATCH']
    released=[r for r in events if r['event']=='RELEASE_FIRST_CLIENT_TOKEN']
    assert len(acquired)==len(released)==len(result['rows'])
    assert {r['request_id'] for r in acquired}=={r['request_id'] for r in released}
    waits=lines(folder/'P_wait_segments.jsonl')
    result['measured_credit_events']=dict(acquired=len(acquired),released=len(released),
        maximum_owned=max(r['owned_before']+1 for r in acquired),
        source_credit_full_P_not_active_wall_ms=1000*sum(r['end']-r['start'] for r in waits if r['reason']=='SOURCE_CREDIT_FULL'),
        scope='Coordinator state intervals, not GPU idle. Direct request blocking and total propagated queue effects are different quantities.')
    result['KV_refusals']={}
    for role in ('prefill','decode'):
        rows=json.loads((folder/f'KV_allocation_refusals_{role}.json').read_text())['rows']
        lifetime=json.loads((folder/f'KV_lifetime_{role}.json').read_text())
        result['KV_refusals'][role]=dict(scheduler_refusal_events=len(rows),
            distinct_requests=len({r['request_id'] for r in rows}),lifetime=lifetime)
    return result


def main():
    groups={}
    fixed=ROOT/'analysis/serial_online_causal_control_v1.json'
    if fixed.exists():groups['Last4_FIXED_OUTPUT_h4']=[r for r in json.loads(fixed.read_text())['results'] if r['causal_mode']=='ONLINE']
    natural=ROOT/'analysis/credit_sweep_v1.json'
    if natural.exists():groups['Last8_NATURAL_OUTPUT_h32']=[r for r in json.loads(natural.read_text())['results'] if r['status']=='MEASURED_ONLINE']
    values={group:[dict(axis=r['axis'],**measure(r['name'])) for r in rows] for group,rows in groups.items()}
    identity=[]
    for axis in ('Native','Layer','Token'):
        arms=[r for r in groups.get('Last8_NATURAL_OUTPUT_h32',[]) if r['axis']==axis]
        if not arms:continue
        reference=sorted(lines(ROOT/'raw'/arms[0]['name']/'requests.jsonl'),key=lambda r:r['sample_id'])
        for row in arms:
            actual=sorted(lines(ROOT/'raw'/row['name']/'requests.jsonl'),key=lambda r:r['sample_id'])
            identity.append(dict(axis=axis,credit=row['credit'],reference=arms[0]['name'],name=row['name'],
                requests=len(actual),prompt_ids_equal=[r['prompt_id'] for r in reference]==[r['prompt_id'] for r in actual],
                strict_output_matches=sum(a['output_ids']==b['output_ids'] for a,b in zip(reference,actual)),
                output_tokens=sum(r['output_tokens'] for r in actual)))
    write(ROOT/'analysis/source_credit_event_decomposition_v1.json',dict(groups=values,within_axis_credit_output_identity=identity,evidence='TRACE_ANALYSIS',
        scope='No serial-barrier arm enters this credit-only accounting. Fixed-output h4 and Natural EOS h32 interventions remain separate. '
        'One independent start per ablation cell. Allocation refusal events are not distinct requests.'))
    print(json.dumps({group:[dict(axis=r['axis'],credit=r['credit'],P_queue_mean_ms=r['summary']['observed_P_queue_ms']['mean'],
        direct_credit_mean_ms=r['summary']['direct_credit_block_ms']['mean'],
        source_chain_mean_ms=r['summary']['previous_source_chain_ms']['mean'],
        credit_full_wall_ms=r['measured_credit_events']['source_credit_full_P_not_active_wall_ms']) for r in rows] for group,rows in values.items()},indent=2))


if __name__=='__main__':main()

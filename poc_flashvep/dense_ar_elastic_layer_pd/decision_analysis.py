"""Audit actual cut commits without calling forced cuts online adaptation."""
from collections import Counter
from .common import *
from .analyze import records,completed_D_layers


TIME_GATES={'KNOWN_DECODE_NOT_DRAINED_BEFORE_EARLIEST_HANDOFF','INSUFFICIENT_KNOWN_INGRESS_WINDOW',
    'INSUFFICIENT_IDLE_PREDICTION','DECODE_BUSY_AT_PREDICTED_HANDOFF','DECODE_DEADLINE_BEFORE_PROPOSED_QUANTUM',
    'BOUNDED_ADMISSION_WAIT_EXCEEDED'}


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    decisions=records(folder/'online_decisions.jsonl');commits=records(folder/'actual_cut_commits.jsonl')
    requests=records(folder/'requests.jsonl');counts=Counter(completed_D_layers(row,cfg) for row in requests)
    committed={r['generation']:r for r in commits}
    selected=Counter();accepted=Counter();rejected=Counter();fallback=Counter();examples=[]
    for row in decisions:
        choice=row['choice'];selected[choice['layers']]+=1
        if not choice['layers']:fallback[choice['reason']]+=1
        for c in choice['candidates']:
            if c['accepted']:accepted[c['layers']]+=1
            for reason in c['rejection_reasons']:rejected[f"Last{c['layers']}:{reason}"]+=1
        if choice['layers'] not in (2,4,8):continue
        actual=committed.get(row['generation'])
        if actual is None or actual['offloaded_layers']!=choice['layers']:continue
        bigger=[c for c in choice['candidates'] if c['layers']>choice['layers']]
        if not bigger:continue
        pure_time=all(TIME_GATES.intersection(c['rejection_reasons']) for c in bigger)
        any_time=any(TIME_GATES.intersection(c['rejection_reasons']) for c in bigger)
        budget_assisted=any(any(r.startswith('MATCHED_') for r in c['rejection_reasons']) for c in bigger)
        examples.append(dict(time=row['time'],chosen_layers=choice['layers'],observation=row['observation'],
            all_larger_cuts_rejected_by_time_estimate=pure_time,at_least_one_larger_time_rejection=any_time,
            experimental_budget_assisted=budget_assisted,candidates=choice['candidates'],
            scope='Actual online prediction and choice. The predicted window is not measured physical idle.'))
    return dict(name=name,policy=cfg.get('policy'),forced_correctness_only=cfg.get('policy')=='FORCED_QUALIFICATION',
        completed_requests=len(requests),completed_cut_histogram=dict(counts),
        actual_migrated_requests=sum(n for m,n in counts.items() if m),
        actual_completed_migrated_layers=sum(m*n for m,n in counts.items()),
        total_prefill_layers=36*len(requests),selected_group_histogram=dict(selected),
        accepted_candidate_groups=dict(accepted),candidate_rejection_counts=dict(rejected),native_fallback_group_reasons=dict(fallback),
        commits=len(commits),smaller_cut_examples=examples,
        pure_predicted_time_fallback_groups=sum(e['all_larger_cuts_rejected_by_time_estimate'] for e in examples),
        budget_assisted_smaller_groups=sum(e['experimental_budget_assisted'] for e in examples),
        evidence='TRACE_ANALYSIS',
        scope='Completed work comes from successful request records. Candidate rejections overlap. Group counts are not request counts. Forced qualification is not an adaptive serving result.')


def main():
    results=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
        if not (folder/'requests.jsonl').exists():continue
        result=analyze(folder.name);results.append(result)
    write(ROOT/'analysis/actual_cut_decision_audit.json',dict(results=results))
    print('DECISION_AUDIT',len(results))


if __name__=='__main__':main()

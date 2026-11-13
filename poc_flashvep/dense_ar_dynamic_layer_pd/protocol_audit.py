"""Audit actual commit order and cohort retirement from preserved Host logs."""
import collections
from .common import *
from .analyze import records,percentile


def distribution(values):
    return dict(samples=len(values),p50=percentile(values,.5),p95=percentile(values,.95),
        p99=percentile(values,.99),maximum=max(values,default=None))


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    req={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
    if cfg['arm']=='DP2':return dict(name=name,status='NOT_APPLICABLE',reason='Colocated has no P-to-D state commit.')
    commits=records(folder/'actual_cut_commits.jsonl')
    sources=records(folder/'actual_submissions.jsonl')
    source_times={'q'+str(r['sample_id']):r['actual_send_start'] for r in sources if 'actual_send_start' in r}
    reserve={r['generation']:r for r in records(folder/'reservation_observations.jsonl') if r['accepted']}
    cuts={};order_failures=[];decision_to_first=[]
    for c in commits:
        if c['offloaded_layers']:
            r=reserve.get(c['generation'])
            if r is None or r['source_receive']>c['time']:order_failures.append(dict(generation=c['generation'],reason='NO_PRIOR_RESERVATION_GRANT'))
        for rid in c['requests']:
            if rid in cuts:order_failures.append(dict(request_id=rid,reason='DUPLICATE_SOURCE_CUT_COMMIT'))
            cuts[rid]=c['cut']
            if rid not in source_times or source_times[rid]<c['time']:
                order_failures.append(dict(request_id=rid,reason='PREFIX_SUBMITTED_BEFORE_CUT_COMMIT'))
            if c['offloaded_layers']:decision_to_first.append(1000*(req[rid]['first_token']-c['time']))
    if commits and set(cuts)!=set(req):order_failures.append(dict(reason='INCOMPLETE_COMMIT_COVERAGE'))
    admissions=records(folder/'D_host_admission.jsonl');cohorts=collections.defaultdict(list)
    for a in admissions:cohorts[a['cohort']].append(a)
    previous_finish=None;overlaps=[];waits=[]
    handoffs={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    for index,values in sorted(cohorts.items()):
        first=min(r['time'] for r in values)
        if previous_finish is not None and first<previous_finish:
            overlaps.append(dict(cohort=index,admission=first,previous_user_finish=previous_finish))
        previous_finish=max(req[r['request_id']]['finish'] for r in values)
        for a in values:
            rid=a['request_id']
            fixed_cut=cfg['layer_cut'] if cfg.get('layer_mode')=='MIGRATE' else 36
            if cuts.get(rid,fixed_cut)<36 and rid in handoffs:
                waits.append(1000*(a['time']-handoffs[rid]))
    return dict(name=name,evidence='TRACE_ANALYSIS_HOST_PROTOCOL',
        committed_request_count=len(cuts),source_commit_order_failures=order_failures,
        source_commit_order_pass=not order_failures if commits else 'NOT_APPLICABLE_FIXED_STATIC_OR_NATIVE',
        prior_Decode_cohort_admission_overlap=overlaps,prior_Decode_cohort_retired_before_admission=not overlaps,
        completed_migrated_request_commit_to_first_ms=distribution(decision_to_first),
        migrated_Source_ready_to_D_host_admission_ms=distribution(waits),
        scope='The audit checks Host protocol order. It does not establish exclusive kernel overlap, wire time, or unknown-future safety. Fixed Static uses its preselected cut without the dynamic reservation protocol.')


def main():
    names=['online_COST_O16_N64_v1','online_DEADLINE_O16_N64_v1','online_IDLE_O16_N64_v1']
    path=ROOT/'analysis/heldout_results.json'
    if path.exists():names+=[r['name'] for r in json.loads(path.read_text())['results']]
    path=ROOT/'analysis/exploratory_fraction50_results.json'
    if path.exists():names+=[r['name'] for r in json.loads(path.read_text())['results']]
    result=[analyze(n) for n in names]
    write(ROOT/'analysis/actual_protocol_audit.json',dict(results=result))
    print('PROTOCOL_AUDIT',len(result))


if __name__=='__main__':main()

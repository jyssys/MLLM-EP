"""Join causal choices to measured native quanta. Keep model utility separate."""
import json
from collections import defaultdict,deque
import numpy as np
from .common import ROOT,write
from .cost_predictor import csv_file
from .optimizer import enumerate_candidates
from .metrics import paired_block_interval

def main():
    from .cost_predictor import ServiceLookup
    artifact=ROOT/'cost_model_artifacts/service_lookup_v1.json'
    lookup=ServiceLookup(json.loads(artifact.read_text()))
    actions=[];opportunities=[];cohorts=[]
    for pattern in ('cb_p3_budget_*','cb_p5_*'):
        for folder in sorted((ROOT/'raw').glob(pattern)):
            if not (folder/'completed.json').exists():
                continue
            for file in folder.glob('*_T*.json'):
                data=json.loads(file.read_text())
                if not data.get('optimizer_decisions'):
                    continue
                issued=defaultdict(deque)
                for q in data['service_observations']:
                    for s in q['spans']:
                        issued[(s['id'],s['start'],s['depth'],s['phase'])].append(q)
                matched=0;errors=[]
                for decision in data['optimizer_decisions']:
                    s=decision['state'];chosen=decision['selected'];key=(s['id'],s['start'],s['depth'],s['phase'])
                    if not issued[key]:
                        continue
                    q=issued[key].popleft();matched+=1
                    predicted=chosen['opening_ms'] if s['phase']=='prefilling_prelude' else chosen['current_ms']
                    over=q['service_ms']>decision['budget_ms'];errors.append(q['service_ms']-predicted)
                    actions.append(dict(run_id=folder.name,policy=data['label'],P=data['P'],N=data['configured_N'],GPU=data['physical_GPU'],
                        trial=data['trial'],phase=s['phase'],state_sha256=decision['state_sha256'],id=s['id'],start=s['start'],depth=s['depth'],
                        C=chosen['C'],G=0 if s['phase']=='prefilling_prelude' else chosen['G'],planned_G=chosen['G'],
                        budget_ms=decision['budget_ms'],predicted_service_ms=predicted,
                        actual_service_ms=q['service_ms'],over_budget=over,known_shape=chosen['known'],
                        fallback='FALLBACK' in decision['reason'],selector_us=decision['selector_ns']/1000,
                        complete_tokens_now=chosen['complete_tokens_now'],predicted_full_residual_ms=chosen['residual_chunk_ms'],
                        request_event_ready=s['request_event_ready'],evidence=data['lane']))
                    if data['label']=='J1' and s['request_event_ready']:
                        joint=enumerate_candidates(s,decision['budget_ms'],lookup,'JOINT',1024,'U2',0.)
                        c_only=enumerate_candidates(s,decision['budget_ms'],lookup,'C_ONLY',1024,'U2',0.)
                        j=max(joint,key=lambda c:c.utility) if joint else None
                        c=max(c_only,key=lambda c:c.utility) if c_only else None
                        opportunities.append(dict(run_id=folder.name,P=data['P'],trial=data['trial'],budget_ms=decision['budget_ms'],
                            kind=s['kind'],depth=s['depth'],joint_feasible=j is not None,C_only_feasible=c is not None,
                            joint_G=None if j is None else j.G,joint_complete_now=None if j is None else j.complete_tokens_now,
                            model_rate_gain_pct=None if j is None or c is None else 100*(j.utility/c.utility-1),
                            scope='MODEL_ESTIMATE_ON_REAL_READY_STATE. Full residual R4 utility, not measured serving gain.'))
                cohorts.append(dict(run_id=folder.name,policy=data['label'],trial=data['trial'],P=data['P'],N=data['configured_N'],
                    matched_actions=matched,issued_prefill_quanta=len(data['service_observations']),
                    selector_decisions=len(data['optimizer_decisions']),MAE_ms=float(np.mean(np.abs(errors))) if errors else None,
                    admission_checks=data.get('admission_cpu',{}).get('checks') if data.get('admission_cpu') else None,
                    admission_total_ms=data.get('admission_cpu',{}).get('total_ns',0)/1e6 if data.get('admission_cpu') else None))
    csv_file(ROOT/'budget_action_trace.csv.gz',actions);csv_file(ROOT/'opportunity_frequency.csv',opportunities)
    csv_file(ROOT/'optimization_overhead.csv',cohorts)
    groups=defaultdict(list)
    for row in actions:
        groups[(row['policy'],row['P'],row['budget_ms'],row['phase'])].append(row)
    summary=[]
    for (policy,P,B,phase),rows in sorted(groups.items()):
        summary.append(dict(policy=policy,P=P,budget_ms=B,path=phase,quanta=len(rows),
            over_budget_pct=100*np.mean([r['over_budget'] for r in rows]),
            fallback_pct=100*np.mean([r['fallback'] for r in rows]),
            unknown_shape_pct=100*np.mean([not r['known_shape'] for r in rows]),
            service_P50_ms=float(np.percentile([r['actual_service_ms'] for r in rows],50)),
            service_P99_ms=float(np.percentile([r['actual_service_ms'] for r in rows],99)),
            selector_P50_us=float(np.percentile([r['selector_us'] for r in rows],50)),
            selector_P99_us=float(np.percentile([r['selector_us'] for r in rows],99)),
            evidence='LIVE_OBSERVER_BUDGET_INTERVENTION_NOT_HARD_SLO'))
    csv_file(ROOT/'budget_overruns.csv',summary)
    write(ROOT/'budget_analysis_summary.json',dict(matched_actions=len(actions),real_ready_snapshots=len(opportunities),cohorts=len(cohorts)))
    print('Budget actions',len(actions),'ready-state model comparisons',len(opportunities))

if __name__=='__main__':
    main()

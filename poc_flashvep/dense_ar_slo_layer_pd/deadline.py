"""Reject suffixes that delay known work beyond an actual SLO deadline."""
from .policy import Choice


def affected_deadline_conflicts(start,finish,work):
    conflicts=[];unaffected=[]
    for item in work:
        # A token deadline from a cohort that retires before this suffix starts
        # is not a deadline for the suffix. The runtime uses this same guard.
        if item.get('retire_upper',float('inf'))<=start:
            unaffected.append(item['request_id']);continue
        ready=item['ready_at'];deadline=item['deadline']
        if ready>=finish:continue
        completion=max(ready,finish)+item.get('service_ms',0.)/1000
        if completion>deadline:
            conflicts.append(dict(request_id=item['request_id'],kind=item['kind'],
                ready_at=ready,deadline=deadline,predicted_completion=completion,
                predicted_lateness_ms=1000*(completion-deadline)))
    return conflicts,unaffected


def direct_choice(policy,state):
    # Reuse the calibrated source, transfer and suffix costs. Replace the old
    # next-P-frontier heuristic with explicit, affected-work deadline tests.
    selected_before=(policy.selected_requests,policy.selected_layer_work)
    old=policy.revised(state)
    policy.selected_requests,policy.selected_layer_work=selected_before
    if not old.candidates:return old
    candidates=[]
    for value in old.candidates:
        c=dict(value);layers=c['layers'];frontier=None;native_frontier=None
        if policy.cfg.get('frontier_completion_model'):
            from .frontier_plan import for_candidate
            frontier=for_candidate(policy,state,layers,c['negotiation_bound_ms'],c['D_available_upper'])
            native_frontier=for_candidate(policy,state,0,0.,state.D_available)
            last=frontier['cohorts'][-1];h=sum(v['handoff_ms'] for v in frontier['cohorts'])
            nh=sum(v['handoff_ms'] for v in native_frontier['cohorts'])
            suffix=sum(v['suffix_ms'] for v in frontier['cohorts']);commit=sum(v['commit_ms'] for v in frontier['cohorts'])
            inter=sum(v['decode_ms'] for v in frontier['cohorts'][:-1])
            upper=frontier['group_first_output']+c['safety_margin_ms']/1000
            personal=1000*(native_frontier['group_first_output']-upper)
            incremental=suffix+commit+inter+max(0.,h-nh)
            c.update(frontier_plan=frontier,Native_frontier_plan=native_frontier,
                predicted_P_cut_ready=frontier['source_ready'],predicted_D_ready=frontier['first_suffix_start'],
                predicted_D_start=frontier['first_suffix_start'],predicted_D_finish_mean=frontier['group_first_output'],
                predicted_D_finish=upper,predicted_Native_finish=native_frontier['group_first_output'],
                predicted_handoff_ms=h,predicted_suffix_ms=suffix,predicted_commit_ms=commit,
                predicted_intercohort_decode_wait_ms=inter,
                predicted_TTFT_saving_ms=personal,predicted_aggregate_gain_ms=personal+c['predicted_queue_saving_ms'],
                predicted_D_incremental_cost_ms=incremental,
                predicted_admission_wait_upper_ms=max(0.,1000*(c['D_available_upper']-
                    min(frontier['source_frontiers'].values()))+policy.costs.get('uncertainty',{}).get('source_absolute_p95_ms',55.)),
                destination_plan=dict(batches=[v['batch'] for v in frontier['cohorts']],
                    handoff_ms=h,suffix_ms=suffix,commit_ms=commit,intercohort_decode_ms=inter,
                    last_cohort_decode_ms=last['decode_ms'],maximum_decode_step_bound_ms=max(
                        frontier['maximum_decode_step_bound_ms'],policy.decode_bound(policy.cfg.get('D_admission_batch',8)))))
            if policy.cfg.get('separate_nominal_objective'):
                nominal_available=state.D_available_nominal if state.D_available_nominal is not None else state.D_available
                nominal=for_candidate(policy,state,layers,c['negotiation_bound_ms'],nominal_available)
                native_nominal=for_candidate(policy,state,0,0.,nominal_available)
                personal_nominal=1000*(native_nominal['group_first_output']-nominal['group_first_output'])
                c.update(nominal_frontier_plan=nominal,Native_nominal_frontier_plan=native_nominal,
                    predicted_personal_nominal_gain_ms=personal_nominal,
                    predicted_D_finish_nominal=nominal['group_first_output'],
                    predicted_D_finish_mean=nominal['group_first_output'],
                    predicted_nominal_aggregate_gain_ms=personal_nominal+c['predicted_queue_saving_ms'],
                    nominal_admission_wait_ms=max(0.,1000*(nominal_available-min(nominal['source_frontiers'].values()))),
                    prediction_scope='The nominal objective excludes explicit uncertainty margins. Its component lookups remain empirical p95 estimates, not an unbiased statistical mean. Safety uses the unchanged upper plan.')
        start=c['predicted_D_start'];finish=c['predicted_D_finish']
        failures=[r for r in c['rejection_reasons'] if r not in (
            'BOUNDED_ADMISSION_WAIT_EXCEEDED','INSUFFICIENT_KNOWN_INGRESS_WINDOW')]
        if frontier:
            failures=[r for r in failures if r!='NO_AGGREGATE_COMPLETION_BENEFIT']
            if c.get('predicted_nominal_aggregate_gain_ms',c['predicted_aggregate_gain_ms'])<=policy.cfg.get('minimum_predicted_benefit_ms',0.):
                failures.append('NO_AGGREGATE_COMPLETION_BENEFIT')
        if state.P_backlog==0 and policy.cfg.get('allow_zero_queue_personal_benefit'):
            if c.get('predicted_personal_nominal_gain_ms',float('-inf'))<=policy.cfg.get('minimum_personal_nominal_gain_ms',0.):
                failures.append('NO_PERSONAL_COMPLETION_BENEFIT_WITHOUT_P_QUEUE')
        affected=[dict(item,ready_at=c['predicted_known_next_ingress']) if item.get('ordering')=='AFTER_CANDIDATE'
            else item for item in state.affected_deadlines]
        recovery=policy.cfg.get('expired_ttft_policy')=='QUEUE_RECOVERY'
        expired_pending=[item for item in affected if recovery and item['kind']=='TTFT' and item['deadline']<=state.now]
        # A past first-token deadline cannot become a protected deadline after
        # this decision. Keep its request in service and in the SLO denominator.
        # Never exempt an ITL deadline or a future first-token deadline.
        affected=[item for item in affected if item not in expired_pending]
        conflicts,retired=affected_deadline_conflicts(start,finish,affected)
        if conflicts:failures.append('AFFECTED_DECODE_OR_FIRST_TOKEN_DEADLINE')
        own_misses=[]
        if frontier and state.own_TTFT_deadlines:
            own_misses=[dict(request_id=state.group_request_ids[i],deadline=deadline,
                predicted_first=frontier['first_outputs'][i]+c['safety_margin_ms']/1000)
                for i,deadline in enumerate(state.own_TTFT_deadlines)
                if frontier['first_outputs'][i]+c['safety_margin_ms']/1000>deadline]
        elif state.own_TTFT_deadline is not None and finish>state.own_TTFT_deadline:
            own_misses=[dict(deadline=state.own_TTFT_deadline,predicted_first=finish)]
        expired_own=[item for item in own_misses if recovery and item['deadline']<=state.now]
        if any(item not in expired_own for item in own_misses):
            failures.append('OWN_TTFT_DEADLINE')
        # Keep an explicit bounded residence limit. This is not an idle budget.
        if c['predicted_admission_wait_upper_ms']>policy.cfg.get('direct_wait_limit_ms',1000.):
            failures.append('DIRECT_BOUNDED_RESIDENCE')
        plan=c.get('destination_plan')
        own_decode=plan['last_cohort_decode_ms'] if plan else policy.decode_bound(state.group_size)*max(0,state.maximum_group_output_tokens-1)
        own_limit=state.own_ITL_limit_ms if state.own_ITL_limit_ms is not None else policy.cfg.get('direct_own_ITL_limit_ms',float('inf'))
        if plan and plan['maximum_decode_step_bound_ms']>own_limit:
            failures.append('OWN_COHORT_DECODE_STEP_BOUND_EXCEEDS_SLO')
        cohort_done=finish+own_decode/1000
        pending_failures=[]
        for item in affected:
            if item.get('ordering')!='AFTER_CANDIDATE':continue
            completion=max(item['ready_at'],cohort_done)+item.get('service_ms',0.)/1000
            if completion>item['deadline']:
                pending_failures.append(dict(request_id=item['request_id'],kind=item['kind'],
                    predicted_completion=completion,deadline=item['deadline']))
        if pending_failures:failures.append('NEXT_INGRESS_DEADLINE_AFTER_COHORT')
        # This empirical surrogate charges both suffix occupancy and the Decode
        # residence of the admitted cohort. It is not a future-goodput oracle.
        occupancy=c['predicted_D_incremental_cost_ms']+own_decode
        score=c['predicted_aggregate_gain_ms']/max(1.,occupancy)
        if 'predicted_nominal_aggregate_gain_ms' in c:score=c['predicted_nominal_aggregate_gain_ms']/max(1.,occupancy)
        batch=state.group_size
        native_plan=policy.destination_plan(0,batch,state.maximum_group_output_tokens) if plan else None
        native_D=(native_plan['handoff_ms']+native_plan['intercohort_decode_ms']+own_decode
            if native_plan else policy._cost('handoff_ms',0,batch)+own_decode)
        native_cycle=max(policy._cost('P_prefix_ms',0,batch),native_D)
        candidate_cycle=max(c['predicted_prefix_ms']+c['negotiation_bound_ms'],
            c['predicted_handoff_ms']+c['predicted_suffix_ms']+c['predicted_commit_ms']+
            c.get('predicted_intercohort_decode_wait_ms',0.)+own_decode)
        candidate_cycle+=c['predicted_admission_wait_upper_ms']
        if frontier:
            native_wait=max(0.,1000*(state.D_available-min(native_frontier['source_frontiers'].values())))
            native_cycle=max(policy._cost('P_prefix_ms',0,batch),native_frontier['service_ms'])+native_wait
            candidate_cycle=max(c['predicted_prefix_ms']+c['negotiation_bound_ms'],frontier['service_ms'])+c['predicted_admission_wait_upper_ms']
            if 'nominal_frontier_plan' in c:
                nominal=c['nominal_frontier_plan'];native_nominal=c['Native_nominal_frontier_plan']
                nominal_available=state.D_available_nominal if state.D_available_nominal is not None else state.D_available
                native_wait=max(0.,1000*(nominal_available-min(native_nominal['source_frontiers'].values())))
                native_cycle=max(policy._cost('P_prefix_ms',0,batch),native_nominal['service_ms'])+native_wait
                candidate_cycle=max(c['predicted_prefix_ms']+c['negotiation_bound_ms'],nominal['service_ms'])+c['nominal_admission_wait_ms']
            if policy.cfg.get('pipeline_objective_excludes_existing_wait'):
                # Existing queue residence remains in every deadline and
                # completion predicate. It is not new GPU service demand.
                native_cycle=max(policy._cost('P_prefix_ms',0,batch),native_frontier['service_ms'])
                candidate_cycle=max(c['predicted_prefix_ms']+c['negotiation_bound_ms'],frontier['service_ms'])
        pipeline_gain=batch*1000*(1/max(1.,candidate_cycle)-1/max(1.,native_cycle))
        if policy.cfg.get('direct_objective')=='PIPELINE_RATE' and pipeline_gain<=0:
            failures.append('NO_PREDICTED_PIPELINE_RATE_BENEFIT')
        c.update(accepted=not failures,rejection_reasons=failures,
            reason=failures[0] if failures else 'DIRECT_AFFECTED_WORK_DEADLINE_PASS',
            affected_deadline_conflicts=conflicts,retired_before_suffix=retired,
            next_ingress_deadline_conflicts=pending_failures,
            predicted_suffix_cohort_retirement=cohort_done,
            own_TTFT_deadline=state.own_TTFT_deadline,
            own_TTFT_deadline_conflicts=own_misses,
            expired_own_TTFT_not_protected=expired_own,
            expired_pending_TTFT_not_protected=expired_pending,
            expired_TTFT_policy=policy.cfg.get('expired_ttft_policy','STRICT'),
            own_ITL_limit_ms=own_limit if own_limit!=float('inf') else None,
            direct_score=score,direct_D_occupancy_ms=occupancy,
            predicted_Native_pipeline_cycle_ms=native_cycle,
            predicted_candidate_pipeline_cycle_ms=candidate_cycle,
            predicted_pipeline_rate_gain_requests_s=pipeline_gain,
            objective_scope=('An empirical incremental-service bottleneck surrogate. Existing waiting remains in all safety and completion tests, not in the new service demand.'
                if policy.cfg.get('pipeline_objective_excludes_existing_wait') else 'An empirical bottleneck-cycle surrogate. It is not measured throughput or a prediction of unknown future goodput.'),
            earliest_existing_decode_deadline=state.earliest_decode_deadline,
            deadline_scope='Only affected work has a direct deadline predicate. Retired-cohort token deadlines are not suffix deadlines. Optional QUEUE_RECOVERY exempts only TTFT deadlines already expired at this decision. Such requests still fail the unchanged SLO test. ITL and future TTFT predicates remain strict. Unknown arrivals are not observed.')
        candidates.append(c)
    legal=[c for c in candidates if c['accepted']]
    if not legal:return Choice(reason='NO_DIRECT_DEADLINE_FEASIBLE_CUT',candidates=candidates)
    if policy.cfg.get('direct_objective')=='PIPELINE_RATE':
        best=max(legal,key=lambda c:(c['predicted_pipeline_rate_gain_requests_s'],c['direct_score'],-c['layers']))
    else:best=max(legal,key=lambda c:(c['direct_score'],c['predicted_aggregate_gain_ms'],-c['layers']))
    policy.selected_requests+=state.group_size;policy.selected_layer_work+=state.group_size*best['layers']
    return Choice(best['layers'],'DIRECT_DEADLINE_OCCUPANCY_AWARE',candidates)

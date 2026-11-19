"""Read only state that existed before a measured P-prefix decision."""


def pressure(decision,current_cost,reference_cost,output,batch):
    observation=decision['observation']
    P_pending=observation.get('P_ready_external',observation['P_backlog'])
    D_ready=len(decision['pending_D_requests'])
    active=decision['active_D_requests']
    delivered=decision['active_delivered_counts']
    current_cohort=current_cost.preparation*batch+current_cost.decode+current_cost.D_gap
    reference_cohort=reference_cost.preparation*batch+reference_cost.decode+reference_cost.D_gap
    if not active:
        residual=0.
    elif min(delivered.get(rid,0) for rid in active)==0:
        residual=current_cohort
    else:
        remaining=max(output-delivered[rid] for rid in active)
        residual=remaining*current_cost.decode/max(output-1,1)
    active_fraction=residual/reference_cohort
    R=(1+D_ready/batch+active_fraction)/(1+P_pending)
    assert min(P_pending,D_ready,R)>=0
    return dict(pressure_ratio=R,P_pending=P_pending,D_ready_requests=D_ready,
        active_D_request_count=len(active),active_D_cohort_count=int(bool(active)),
        calibrated_active_remaining_seconds=residual,
        active_fraction=active_fraction,
        state_age_ms=observation.get('state_age_ms'),
        scope='A host-state feature at P decision time. The calibrated residual is not a future measured completion. Existing D_available and predicted_active_D_finish are deliberately ignored. No physical GPU idle follows.')


def scores(R,table,kappa,batch):
    # Multiplying both scores by1+P_pending does not change the selected axis.
    return {a:c.P+c.P_gap+kappa*R*(c.preparation+(c.decode+c.D_gap)/batch) for a,c in table.items()}

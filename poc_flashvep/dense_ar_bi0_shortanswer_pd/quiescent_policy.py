"""Fail closed when a quiescent reservation lacks a causal certificate."""


def rejection_reasons(D_active,D_pending,source_outstanding,P_waiting,minimum=1):
    reasons=[]
    if D_active is not None:reasons.append('D_ACTIVE')
    if D_pending:reasons.append('D_READY_PENDING')
    if source_outstanding:reasons.append('OTHER_SOURCE_WORK_CAN_BECOME_READY')
    if P_waiting<minimum:reasons.append('NO_P_BACKLOG_BEHIND_CURRENT')
    return reasons

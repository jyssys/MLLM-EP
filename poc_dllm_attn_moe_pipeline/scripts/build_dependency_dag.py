"""Critical-path tools with no duplicated savings or implicit free collectives."""


def critical_path(nodes):
    finish = {}
    remaining = {n['node_id']:n for n in nodes}
    while remaining:
        ready = [key for key,n in remaining.items() if all(p in finish for p in n['predecessors'])]
        if not ready:
            raise ValueError('Cyclic or missing dependency')
        for key in ready:
            n = remaining.pop(key)
            if n['duration_us'] is None:
                raise ValueError('Unknown cost cannot be zero')
            finish[key] = max((finish[p] for p in n['predecessors']), default=0.) + n['duration_us']
    return max(finish.values(),default=0.),finish


def boundary_dag(producer_groups, local_work, current_work, merge_work, final_work, overlap):
    """Producer costs already include real shared/add/all-gather publication.

    Producer collective order is globally fixed. Current-block attention waits
    for EVERY group's fresh KV. Resource contention omitted only in envelope.
    """
    nodes = []
    for g,p in enumerate(producer_groups):
        nodes.append(dict(node_id=f'p{g}',predecessors=[] if g==0 else [f'p{g-1}'],duration_us=p))
        nodes.append(dict(node_id=f'q{g}',predecessors=[f'p{g}' if overlap else f'p{len(producer_groups)-1}'],duration_us=local_work[g]))
    nodes.append(dict(node_id='current',predecessors=[f'q{g}' for g in range(len(producer_groups))],duration_us=current_work))
    nodes.append(dict(node_id='merge',predecessors=['current'],duration_us=merge_work))
    nodes.append(dict(node_id='final',predecessors=['merge'],duration_us=final_work))
    return nodes

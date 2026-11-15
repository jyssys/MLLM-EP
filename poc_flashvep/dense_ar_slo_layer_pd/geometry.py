"""Select complete prompt rows without exceeding the real token budget."""


def prefix_group(indices,lengths,maximum,row_budget,equal_lengths=False):
    count=0;rows=0;first=None
    for index in list(indices)[:maximum]:
        length=lengths[index]
        if first is None:first=length
        if equal_lengths and length!=first:break
        if rows+length>row_budget:break
        rows+=length;count+=1
    assert count,'The first full prompt must fit the token budget'
    return count


def macro_frontier_ready(members,states):
    # A later destination cohort must not wait for earlier members to become
    # pending again. Their states advance monotonically to decode/finished.
    return all(states.get(rid) in ('D_HOST_PENDING','decode','finished') for rid in members)


def source_queued_inside_group(count,slots):
    return max(0,count-slots)

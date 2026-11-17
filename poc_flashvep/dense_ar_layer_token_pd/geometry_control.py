"""Replay declared FIFO group membership without reading future timing."""
def group_size(index,cfg):
    plan=cfg.get('planned_source_group_sizes')
    if plan is None:
        return cfg.get('P_group_size',cfg['P_slots'])
    start=0
    for size in plan:
        assert 0<size<=cfg.get('P_group_size',cfg['P_slots'])
        if index==start:
            return size
        start+=size
    raise AssertionError('Source index is not a declared group boundary')

"""Rejected debug hypothesis. This CPU-only planner is not installed.

The audited packed RecurrentGraphs already uses exact query-count keys.
The initial bucket hypothesis confused it with SteadyRecurrentGraphs.
"""


def preferred(cfg, role, old, limit, budget):
    t = cfg['length']
    frontier = cfg.get('native_frontier')
    cut = cfg.get('native_partial_cut')
    if role == 'prefill':
        regions = ([frontier[1]] if frontier and frontier[0] == 0 else
                   [t, frontier[1]] if frontier else [t])
        targets = {n * b for n in regions for b in (1, 2) if n * b <= limit}
    elif frontier or cut is not None:
        suffix = t - frontier[1] if frontier else t
        targets = {suffix * b for b in (1, 2, 4, 8) if suffix * b <= min(limit, cfg['D_token_budget'])}
    else:
        return tuple(old)
    if len(targets) > budget:
        raise ValueError('Exact candidate shapes exceed the shared graph budget')
    for size in sorted(old, reverse=True):
        if len(targets) < budget:
            targets.add(size)
    return tuple(sorted(targets))

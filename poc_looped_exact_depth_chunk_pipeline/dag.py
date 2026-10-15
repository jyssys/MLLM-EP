"""Exact macro dependencies and their tied-unrolled PP isomorphism."""
import heapq


def tasks(chunks,depth=4):
    return [(r,j) for r in range(depth) for j in range(chunks)]


def predecessors(r,j):
    return ([(r-1,j)] if r else [])+([(r,j-1)] if j else [])


def diagonal_order(chunks,depth=4):
    return sorted(tasks(chunks,depth),key=lambda x:(sum(x),x[0]))


def verify_order(order):
    completed=set()
    for task in order:
        assert task not in completed
        assert set(predecessors(*task))<=completed,(task,'missing hidden or same-depth prefix KV')
        completed.add(task)
    return True


def micro_isomorphism(chunks,depth=4,layers=48):
    """Retain separate KV production/publication and native pass norms."""
    recurrent=set();generic=set()
    def edge(a,b):recurrent.add((a,b))
    def mapped(node):
        r,l,j,kind=node
        if kind=='EMB':return (-1,j,kind)
        if kind=='NORM':return (r*layers+layers-1,j,'NORM_AFTER')
        return (r*layers+l,j,kind)
    for r in range(depth):
        for j in range(chunks):
            for l in range(layers):
                incoming=(r,l-1,j,'OUT') if l else ((r-1,layers,j,'NORM') if r else (-1,0,j,'EMB'))
                key=(r,l,j,'KV');pub=(r,l,j,'PUB');out=(r,l,j,'OUT')
                edge(incoming,key);edge(key,pub);edge(pub,out)
                if j:edge((r,l,j-1,'PUB'),pub)
                if l==layers-1:edge(out,(r,layers,j,'NORM'))
    for d in range(depth*layers):
        for j in range(chunks):
            incoming=(-1,j,'EMB') if d==0 else (d-1,j,'NORM_AFTER' if d%layers==0 else 'OUT')
            generic.update([(incoming,(d,j,'KV')),((d,j,'KV'),(d,j,'PUB')),((d,j,'PUB'),(d,j,'OUT'))])
            if j:generic.add(((d,j-1,'PUB'),(d,j,'PUB')))
            if d%layers==layers-1:generic.add(((d,j,'OUT'),(d,j,'NORM_AFTER')))
    relabeled={(mapped(a),mapped(b)) for a,b in recurrent}
    assert relabeled==generic,'Native norm or KV-frontier dependency changed'
    return dict(edges=len(generic),nodes=len({n for pair in generic for n in pair}),edge_sets_equal=True)


def schedule(costs,transfer_ms,stages=4):
    """Resource-limited exact macro schedule. Durations are modeled inputs."""
    chunks=len(costs);depth=4;ends={};available=[0.]*stages;rows=[]
    per_stage=depth//stages
    # Prefer the earliest chunk through both colocated passes. A diagonal list
    # over individual loops would delay PP2's first downstream publication.
    for j in range(chunks):
        for owner in range(stages):
            ready=ends[(owner-1,j)]+transfer_ms[j] if owner else 0.
            start=max(ready,available[owner])
            for offset in range(per_stage):
                end=start+costs[j]
                rows.append(dict(loop=owner*per_stage+offset,chunk=j,stage=owner,start_ms=start,end_ms=end,compute_ms=costs[j]))
                start=end
            ends[(owner,j)]=end;available[owner]=end
    return max(ends.values()),rows

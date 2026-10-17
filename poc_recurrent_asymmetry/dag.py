"""Exact dependency graphs and resource-limited offline schedules."""


def micro_graph(chunks,depth=4,layers=24):
    graph={}
    for r in range(depth):
        for l in range(layers):
            for c in range(chunks):
                incoming=(r,l-1,c,'O') if l else ((r-1,layers-1,c,'N') if r else None)
                q=(r,l,c,'Q');k=(r,l,c,'K');out=(r,l,c,'O')
                graph[q]=set() if incoming is None else {incoming}
                graph[k]={q}|({(r,l,c-1,'K')} if c else set())
                graph[out]={k}|({(r,l,c-1,'K')} if c else set())
                if l==layers-1:graph[r,l,c,'N']={out}
    return graph


def ready_set(graph,completed):
    return {node for node,deps in graph.items() if node not in completed and deps<=completed}


def validate_order(graph,order):
    complete=set()
    for node in order:
        assert node not in complete and graph[node]<=complete,(node,'Dependency not ready')
        complete.add(node)
    assert complete==set(graph)
    return True


def isomorphism(chunks,depth=4,layers=24):
    graph=micro_graph(chunks,depth,layers)
    def mapping(node):
        r,l,c,kind=node
        return r*layers+l,c,kind
    mapped={mapping(node):{mapping(dep) for dep in deps} for node,deps in graph.items()}
    generic={}
    for d in range(depth*layers):
        for c in range(chunks):
            incoming=(d-1,c,'N' if d%layers==0 else 'O') if d else None
            generic[d,c,'Q']=set() if incoming is None else {incoming}
            generic[d,c,'K']={(d,c,'Q')}|({(d,c-1,'K')} if c else set())
            generic[d,c,'O']={(d,c,'K')}|({(d,c-1,'K')} if c else set())
            if d%layers==layers-1:generic[d,c,'N']={(d,c,'O')}
    assert mapped==generic
    return dict(nodes=len(graph),edges=sum(map(len,graph.values())),edge_sets_equal=True,
        mapping='logical_layer = recurrent_depth * physical_layers + physical_layer',
        retained='Per-pass terminal norm and separate depth/layer KV publication.')


def schedule(costs,placement,comm_ms=0.,policy='asap',conservative=False):
    """Coarse six-layer tasks. Previous-chunk full output is conservative KV readiness."""
    nodes=set(costs)
    groups=max(t[1] for t in nodes)+1
    deps={t:({(t[0],t[1]-1,t[2])} if t[1] else ({(t[0]-1,groups-1,t[2])} if t[0] else set()))
          |({(t[0],t[1],t[2]-1)} if t[2] else set()) for t in nodes}
    complete={};devices=[0.]*4;links={};timeline=[]
    while len(complete)<len(nodes):
        ready=[t for t in nodes-complete.keys() if deps[t]<=complete.keys()]
        assert ready,'Cyclic or incomplete DAG'
        options=[]
        for t in ready:
            owner=placement(t);release=0.;moves=[]
            # Previous-chunk state stays in the same layer owner. Only incoming hidden edges transfer.
            for parent in deps[t]:
                source=placement(parent);end=complete[parent]
                if source!=owner:
                    edge=(source,owner);begin=max(end,links.get(edge,0.))
                    if conservative:begin=max(begin,devices[source],devices[owner])
                    duration=comm_ms if not callable(comm_ms) else comm_ms(t,source,owner)
                    end=begin+duration;moves.append((edge,begin,end))
                release=max(release,end)
            start=max(release,devices[owner])
            priority=(t[2],t[0],t[1]) if policy=='chunk_fifo' else (-t[0],t[2],t[1])
            options.append((start,priority,t,moves))
        start,_,task,moves=min(options)
        owner=placement(task)
        for edge,begin,end in moves:
            links[edge]=end
            if conservative:
                devices[edge[0]]=max(devices[edge[0]],end);devices[edge[1]]=max(devices[edge[1]],end)
            timeline.append(dict(kind='TRANSFER',source=edge[0],device=edge[1],task=list(task),start_ms=begin,end_ms=end))
        start=max(start,devices[owner]);end=start+costs[task]
        assert all(complete[parent]<=start for parent in deps[task])
        complete[task]=end;devices[owner]=end
        timeline.append(dict(kind='COMPUTE',device=owner,task=list(task),start_ms=start,end_ms=end))
    # Every modeled compute task executes once. No device runs overlapping compute.
    for device in range(4):
        spans=sorted((r['start_ms'],r['end_ms']) for r in timeline if r['kind']=='COMPUTE' and r['device']==device)
        assert all(b<=c+1e-9 for (a,b),(c,d) in zip(spans,spans[1:]))
    return max(complete.values()),timeline

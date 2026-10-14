"""Non-overlap accounting for measured intervals. Unknowns remain explicit."""


def union_ms(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<=start:continue
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    return sum(end-start for start,end in merged)


def exclusive_budget(spans,start,end,start_key='host_start_ns',end_key='host_end_ns'):
    """Assign each segment to its deepest containing scope, never sum parents."""
    boundaries={start,end}
    chosen=[]
    by_id={s['id']:s for s in spans}
    for span in spans:
        a=max(start,span[start_key]);b=min(end,span[end_key])
        if b<=a:continue
        depth=0;parent=span['parent']
        while parent is not None:
            depth+=1;parent=by_id[parent]['parent']
        chosen.append((a,b,depth,span));boundaries.update([a,b])
    out=[];points=sorted(boundaries)
    for a,b in zip(points,points[1:]):
        active=[s for x,y,d,s in chosen if x<=a and y>=b]
        winner=max(active,key=lambda s:next(d for x,y,d,r in chosen if r is s)) if active else None
        out.append(dict(start=a,end=b,kind='outside_scopes' if winner is None else winner['kind'],
            span_id=None if winner is None else winner['id']))
    return out


def kernel_partition(kernels,start,end):
    """Exclusive rank activity is not automatically a dependency critical path."""
    changes={start:[],end:[]}
    for index,k in enumerate(kernels):
        a=max(start,k['start']);b=min(end,k['end'])
        if b<=a:continue
        changes.setdefault(a,[]).append((1,index))
        changes.setdefault(b,[]).append((-1,index))
    active=set();last=start;result=[]
    for position,updates in sorted(changes.items()):
        if position>last:
            categories={kernels[i]['category'] for i in active}
            label='GPU_IDLE_UNATTRIBUTED' if not active else (next(iter(categories)) if len(categories)==1 else 'OVERLAP_MULTIPLE_CATEGORIES')
            result.append(dict(start=last,end=position,category=label,active_count=len(active)))
        for delta,index in updates:
            if delta==1:active.add(index)
            else:active.discard(index)
        last=position
    return result

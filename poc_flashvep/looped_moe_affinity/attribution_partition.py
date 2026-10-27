"""Partition nested critical-stream ranges. Never sum parent and child durations."""
from collections import defaultdict
import heapq
import math

def partition(intervals,window_ms):
    if not math.isfinite(window_ms) or window_ms<=0:
        raise ValueError('The attribution window must be finite and positive.')
    points=[];regions={}
    for i,(start,end,label,*extra) in enumerate(intervals):
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError('Attribution timestamps must be finite.')
        if end<start:
            raise ValueError('An attribution region cannot end before it starts.')
        full_duration=end-start
        start=max(0.,start);end=min(float(window_ms),end)
        if end<=start:continue
        depth=int(extra[2]) if len(extra)>=3 else 0
        regions[i]=(-depth,full_duration,label)
        points.extend([(start,1,i),(end,0,i)])
    points.sort();active=set();heap=[];totals=defaultdict(float);last=0.
    for stamp,kind,index in points+[(float(window_ms),0,-1)]:
        while heap and heap[0][3] not in active:heapq.heappop(heap)
        label=heap[0][2] if heap else 'unattributed'
        if stamp>last:totals[label]+=stamp-last
        if index>=0:
            if kind:
                active.add(index);depth,duration,label=regions[index];heapq.heappush(heap,(depth,duration,label,index))
            else:active.discard(index)
        last=stamp
    assert abs(sum(totals.values())-window_ms)<1e-5
    return dict(totals)

def split_phases(intervals,window_ms,ttft_ms):
    """Use the first output-completion marker, not a sum of forward durations."""
    if not math.isfinite(ttft_ms) or not 0<ttft_ms<window_ms:
        raise ValueError('The first-token boundary must be inside the rollout.')
    remaining=[(start-ttft_ms,end-ttft_ms,label,*extra) for start,end,label,*extra in intervals]
    return {
        'prefill_to_first_output':dict(window_ms=ttft_ms,components_ms=partition(intervals,ttft_ms)),
        'remaining_decode_outputs':dict(window_ms=window_ms-ttft_ms,
            components_ms=partition(remaining,window_ms-ttft_ms)),
    }

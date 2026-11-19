"""Model the reservation reply boundary, not only an average source gap."""
import heapq
import math


def next_boundary(now,cohort,rows,costs,output):
    if not cohort:return now
    start=rows[cohort[0]]['D_start']
    boundary=start
    for i in cohort:
        boundary+=costs[i].preparation
        if now<boundary-1e-12:return boundary
    first=boundary
    finish=rows[cohort[0]]['scheduled_finish']
    if now>=finish-1e-12:return now
    step=(finish-first)/max(1,output-1)
    if step<=0:return now
    k=max(1,math.ceil((now-first+1e-12)/step))
    return min(finish,first+k*step)


def simulate(arrivals,costs,batch=1,window=32,choose=None,kv_slots=None,*,output=16,idle_RTT=.001):
    assert len(arrivals)==len(costs) and len(arrivals)%batch==0 and window>=batch
    assert all(y>=x for x,y in zip(arrivals,arrivals[1:])) and idle_RTT>=0
    costs=list(costs)
    for c in costs:c.validate()
    if kv_slots is not None:assert kv_slots>=batch
    n=len(arrivals);events=[]
    for i,t in enumerate(arrivals):heapq.heappush(events,(t,0,i))
    pending=[];ready=[];cohort=[]
    rows=[dict(arrival=t) for t in arrivals]
    P_busy=False;outstanding=reserved=0
    while events:
        now,kind,index=heapq.heappop(events)
        if kind==0:pending.append(index)
        elif kind==1:
            P_busy=False;rows[index]['P_ready']=now;outstanding+=1;ready.append(index)
        elif kind==2:
            for i in cohort:rows[i]['finish']=now
            outstanding-=len(cohort);reserved-=len(cohort);cohort=[]
        elif kind==3:
            # Model request arrival at D after half the idle round trip.
            boundary=next_boundary(now,cohort,rows,costs,output)
            heapq.heappush(events,(boundary+idle_RTT/2,4,index))
        elif kind==4:
            reserved+=1
            assert kv_slots is None or reserved<=kv_slots
            start=now+costs[index].P_gap
            rows[index].update(P_start=start,reservation_ACK=now,
                reservation_wait=now-rows[index]['reservation_requested'])
            heapq.heappush(events,(start+costs[index].P,1,index))
        if not cohort and len(ready)>=batch:
            cohort=ready[:batch];del ready[:batch]
            start=now+max(costs[i].D_gap for i in cohort)
            first=start+sum(costs[i].preparation for i in cohort)
            finish=first+max(costs[i].decode for i in cohort)
            for i in cohort:rows[i].update(D_start=start,first=first,cohort=tuple(cohort),scheduled_finish=finish)
            heapq.heappush(events,(finish,2,cohort[0]))
        if not P_busy and pending and outstanding<window and (kv_slots is None or reserved<kv_slots):
            i=pending.pop(0);P_busy=True
            if choose is not None:
                finish=rows[cohort[0]]['scheduled_finish'] if cohort else now
                costs[i]=choose(i,now,dict(pending_source_indices=tuple(pending),ready_indices=tuple(ready),
                    active_indices=tuple(cohort),estimated_active_D_remaining=max(0.,finish-now),
                    outstanding=outstanding,reserved_KV_slots=reserved,
                    estimated_D_next_control_boundary=next_boundary(now,cohort,rows,costs,output)))
                costs[i].validate()
            rows[i].update(reservation_requested=now,P_pending_at_P_start=len(pending),
                D_pending_at_P_start=len(ready))
            heapq.heappush(events,(now+idle_RTT/2,3,i))
    assert not pending and not ready and not cohort and outstanding==reserved==0
    assert all('finish' in r for r in rows)
    for r in rows:
        r.update(TTFT=r['first']-r['arrival'],P_wait=r['P_start']-r['arrival'],
            Ready_D_wait=r['D_start']-r['P_ready'])
        assert min(r['TTFT'],r['P_wait'],r['Ready_D_wait'])>=-1e-10
    return rows

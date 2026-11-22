"""Validate a state-updating C0 queue model before any oracle claim."""
import collections
import heapq
import json
import math
import itertools
import numpy as np
from .common import ROOT,write
from .geometry import Partition,aligned_suffix
from .service_accounting import analyze


def nnls(matrix,target):
    """Enumerate feasible least-squares faces for these at-most-seven features."""
    matrix=np.asarray(matrix,float);target=np.asarray(target,float)
    columns=matrix.shape[1];assert columns<=7
    best=np.zeros(columns);loss=float(target@target)
    for size in range(1,columns+1):
        for chosen in itertools.combinations(range(columns),size):
            value=np.linalg.lstsq(matrix[:,chosen],target,rcond=None)[0]
            if np.min(value)<-1e-9:continue
            candidate=np.zeros(columns);candidate[list(chosen)]=np.maximum(value,0)
            error=target-matrix@candidate;current=float(error@error)
            if current<loss:best,loss=candidate,current
    return best,math.sqrt(loss)


def part(cfg,index,length):
    rid='q'+str(index);action=cfg.get('axis_actions_by_request',{}).get(rid,{})
    kind=action.get('kind',cfg.get('axis_kind','Native'))
    m=action.get('cut',cfg.get('axis_cut',0));s=0
    if kind=='Token':s=aligned_suffix(length,m);m=0
    elif kind=='Hybrid':
        f=action.get('suffix_fraction',cfg.get('hybrid_suffix_fraction',0))
        s=0 if not f else min((length-1)//16*16,max(16,int(round(length*f/16))*16))
    elif kind=='Native':m=0
    s=int(cfg.get('suffix_by_request',{}).get(rid,s))
    return Partition(length,m,s)


def features(p,role):
    if role=='P':
        return [1,p.P_rows/1e5,p.P_attention_pairs/1e9,
            math.ceil(p.prefix/8192)*p.lower/100,p.record()['hidden_residual_bytes']/1e8,float(p.upper>0)]
    calls=p.upper*math.ceil(p.prefix/8192)+36*math.ceil(p.suffix/8192)
    if p.upper==p.suffix==0:calls=36
    return [1,p.D_rows/1e5,p.D_attention_pairs/1e9,calls/100,
        p.P_rows/1e5,p.record()['hidden_residual_bytes']/1e8,float(p.upper>0)]


def service(name):
    path=ROOT/'analysis'/(name+'_service_accounting.json')
    return json.loads(path.read_text()) if path.exists() else analyze(name)


def simulate(arrivals,lengths,parts,output,model):
    events=[(float(t),0,i) for i,t in enumerate(arrivals)];heapq.heapify(events)
    pending=collections.deque();ready=collections.deque();owned=set()
    pactive=dactive=None;rows=[{} for _ in arrivals]
    while events:
        t,event,i=heapq.heappop(events)
        if event==0:pending.append(i)
        elif event==1:
            assert pactive==i;pactive=None;ready.append(i);rows[i]['P_ready']=t
        elif event==2:
            assert i in owned;owned.remove(i);rows[i]['first']=t
        elif event==3:
            assert dactive==i;dactive=None;rows[i]['finish']=t
        if dactive is None and ready:
            j=ready.popleft();dactive=j;rows[j]['D_start']=t
            first=max(.001,float(np.dot(features(parts[j],'D'),model['D'])))/1000
            step=max(.001,float(np.dot([1,lengths[j]/1e4],model['decode_step'])))/1000
            heapq.heappush(events,(t+first,2,j));heapq.heappush(events,(t+first+(output-1)*step,3,j))
        if pactive is None and pending and len(owned)<4:
            j=pending.popleft();pactive=j;owned.add(j);rows[j]['P_start']=t
            cost=max(.001,float(np.dot(features(parts[j],'P'),model['P'])))/1000
            heapq.heappush(events,(t+cost,1,j))
        assert len(owned)<=4
    assert all('finish' in r and 'first' in r for r in rows)
    ttft=[1000*(r['first']-a) for r,a in zip(rows,arrivals)]
    return dict(TTFT_p95_ms=float(np.percentile(ttft,95)),TTFT_mean_ms=float(np.mean(ttft)),
        mean_P_queue_ms=float(np.mean([1000*(r['P_start']-a) for r,a in zip(rows,arrivals)])),
        mean_D_ready_wait_ms=float(np.mean([1000*(r['D_start']-r['P_ready']) for r in rows])),
        rows=rows,scope='OFFLINE_REPLAY. P1/D1 whole-request service with credit4 released at D first output. '
            'The event loop updates queues after every action. No frozen-state per-request minimum is used. '
            'This model does not represent C3 mixed batching or measured GPU-exclusive time.')


def main():
    contract=dict(version=1,training='Three serial208-action starts plus the16 Native serial pilots.',
        validation='Every available held-out64 C0 confirmation cell/start. No validation sample is used for fitting.',
        allowed_median_absolute_TTFT_percentage_error=10.,allowed_p90_absolute_TTFT_percentage_error=25.,
        model='Nonnegative host-service regression. Static features are normalized rows,pairs,layer calls,hidden bytes and transport class.',
        no_causal_coefficient_claim=True,
        safety_scope='No validated ITL-tail or joint-SLO model. This model cannot authorize online insertion or certify a SLO-goodput oracle.',
        evidence='FROZEN_OFFLINE_REPLAY_VALIDATION_DESIGN')
    path=ROOT/'configs/c0_queue_model_validation_v1.json'
    if path.exists():assert json.loads(path.read_text())==contract
    else:write(path,contract)
    training=[f'HybridFamily_CostLUT_Serial208_Start{i}_v1' for i in range(3)]
    training += [f'CapacityC0_I{b}_O{o}_Serial16_v1' for b in range(4) for o in (8,16,64,128)]
    x={'P':[],'D':[]};y={'P':[],'D':[]};dx=[];dy=[]
    for name in training:
        cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
        for row in service(name)['rows']:
            i=int(row['request_id'][1:]);p=part(cfg,i,row['input_tokens'])
            for role,target in (('P','P_submit_to_ready_ms'),('D','D_submit_to_accept_ms')):
                x[role].append(features(p,role))
                value=row['P_submit_to_ready_ms'] if role=='P' else sum(row[k] for k in
                    ('D_submit_to_accept_ms','D_accept_to_prepare_ms','D_prepare_to_first_worker_ms'))
                y[role].append(value)
            if row['output_tokens']>1:
                dx.append([1,row['input_tokens']/1e4]);dy.append(row['D_decode_residency_ms']/(row['output_tokens']-1))
    model={role:nnls(np.array(x[role]),np.array(y[role]))[0].tolist() for role in x}
    model['decode_step']=nnls(np.array(dx),np.array(dy))[0].tolist()
    write(ROOT/'analysis/c0_host_service_model_v1.json',dict(model=model,training=training,
        training_observations=len(dx),evidence='OFFLINE_MODEL',
        scope='Coefficients predict observed host service,not exclusive GPU kernels. '
            'Correlated static features make coefficients unsuitable for causal operator attribution. '
            'First-use compilation and transport variance remain in the calibration observations.'))
    protocol=json.loads((ROOT/'configs/heldout64_confirmation_protocol_r2.json').read_text())
    names=sorted({cell[axis] for cell in protocol['cells'] for axis in ('Native','Layer','Token')})
    rows=[]
    for name in names:
        folder=ROOT/'raw'/name
        if not (folder/'dispatcher_result.json').exists():continue
        assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
        cfg=json.loads((folder/'configuration.json').read_text())
        data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
        pred=simulate(cfg['arrival_offsets'],[r['input_tokens'] for r in data],
            [part(cfg,i,r['input_tokens']) for i,r in enumerate(data)],cfg['output_cap'],model)
        actual=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
        error=100*(pred['TTFT_p95_ms']/actual['TTFT_p95_ms']-1)
        rows.append(dict(name=name,measured_TTFT_p95_ms=actual['TTFT_p95_ms'],predicted_TTFT_p95_ms=pred['TTFT_p95_ms'],
            signed_TTFT_percentage_error=error,absolute_TTFT_percentage_error=abs(error),
            predicted_P_queue_ms=pred['mean_P_queue_ms'],predicted_D_ready_wait_ms=pred['mean_D_ready_wait_ms']))
    errors=[r['absolute_TTFT_percentage_error'] for r in rows]
    median=float(np.median(errors));p90=float(np.percentile(errors,90))
    passed=median<=10 and p90<=25
    value=dict(model=model,rows=rows,heldout_server_starts=len(rows),
        median_absolute_TTFT_percentage_error=median,p90_absolute_TTFT_percentage_error=p90,
        TTFT_replay_gate_passed=passed,SLO_oracle_gate_passed=False,online_admission_allowed=False,
        evidence='OFFLINE_REPLAY',
        scope='Held-out request shapes and arrival seeds. The TTFT gate does not qualify ITL tails,joint SLO or online safety. '
            'No oracle serving-speedup claim is permitted from this validation. Native controls are not duplicated across cuts.')
    write(ROOT/'analysis/c0_queue_model_heldout_validation_v1.json',value)
    print('STATE_UPDATING_REPLAY',len(rows),'TTFT gate',passed,'median APE',median,'p90 APE',p90,
        'SLO oracle NOT_QUALIFIED',flush=True)


if __name__=='__main__':main()

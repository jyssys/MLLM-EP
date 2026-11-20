"""Conditional FIFO replay. A finite beam is not a global upper bound."""
import hashlib
import json
import numpy as np
from .common import ROOT,write,digest
from .service_model import predict,error_summary
from .tandem_replay import replay
from .qualification import records


def action_costs(model,row,output_tokens,decode_step):
    costs={}
    for action,value in model['actions'].items():
        curves=value['curves']
        source_curve=curves.get('P_serial_submit_to_ready_ms',curves['P_prepare_to_ready_ms'])
        P=predict(source_curve['coefficients_ms'],row['input_tokens'])/1000
        F=predict(curves['ready_to_first_worker_output_ms']['coefficients_ms'],row['input_tokens'])/1000
        costs[action]=(P,F,decode_step*max(0,output_tokens-1))
    return costs


def metrics(states,arrivals,outputs):
    duration=max(r['D_finish'] for r in states)-min(arrivals)
    ttft=np.array([r['D_first']-a for r,a in zip(states,arrivals)])
    return dict(TTFT_p95_ms=float(np.percentile(ttft,95)*1000),
        TTFT_mean_ms=float(ttft.mean()*1000),
        modeled_output_tokens_per_second=float(sum(outputs)/duration),
        modeled_TTFT_qualified_tokens_per_second=float(sum(o for o,t in zip(outputs,ttft) if t<=5.)/duration),
        TTFT_attainment=float(np.mean(ttft<=5.)),measurement_seconds=float(duration))


def fixed_trace(arrivals,costs,actions,credit=4):
    parts=[costs[i][a] for i,a in enumerate(actions)]
    return replay(arrivals,[p[0] for p in parts],[p[1] for p in parts],[p[2] for p in parts],credit)


def KV_contract_accounting(states,rows,outputs):
    # Qwen3-8B:36 layers,8 KV heads,128 head dimension,K+V,BF16.
    block_tokens=128;block_bytes=36*2*8*128*2*block_tokens
    events=[]
    for state,row,count in zip(states,rows,outputs):
        source_blocks=(row['input_tokens']+1+block_tokens-1)//block_tokens
        dest_blocks=(row['input_tokens']+count+1+block_tokens-1)//block_tokens
        events.extend(((state['P_start'],1,'source',source_blocks),
            (state['D_first'],0,'source',-source_blocks),
            (state['D_start'],1,'destination',dest_blocks),
            (state['D_finish'],0,'destination',-dest_blocks)))
    current=dict(source=0,destination=0);maximum=dict(current)
    for _,_,role,delta in sorted(events):
        current[role]+=delta;assert current[role]>=0
        maximum[role]=max(maximum[role],current[role])
    assert not any(current.values())
    values={role:count*block_bytes for role,count in maximum.items()}
    assert all(value<=42*1024**3 for value in values.values())
    return dict(maximum_blocks=maximum,maximum_KV_bytes=values,block_tokens=block_tokens,block_bytes=block_bytes,
        source_release='D first output',destination_release='D finish',credits_restored=True,
        evidence='STRUCTURAL_ACCOUNTING',scope='Conservative full36 source reservation even for partial actions. '
        'Includes one lookahead token. Queue actions update ownership and KV block totals. '
        'The fixed source-credit4/D1 contract keeps KV below42GiB. This is not a fragmented allocator replay, '
        'a measured memory peak, hidden-buffer memory accounting or a GPU drain certificate.')


def extend(states,index,arrival,cost,credit=4):
    P,F,D=cost;previous=states[-1] if states else None
    release=states[index-credit]['D_first'] if index>=credit else 0.
    ps=max(arrival,previous['P_finish'] if previous else 0.,release);pf=ps+P
    ds=max(pf,previous['D_finish'] if previous else 0.)
    return dict(index=index,P_start=ps,P_finish=pf,D_start=ds,D_first=ds+F,D_finish=ds+F+D)


def beam(arrivals,costs,outputs,allowed,width=24):
    paths=[([],[],0.,0.)]
    for i,arrival in enumerate(arrivals):
        next_paths=[]
        for states,actions,good,total_ttft in paths:
            for action in allowed:
                state=extend(states,i,arrival,costs[i][action]);ttft=state['D_first']-arrival
                new_good=good+(outputs[i] if ttft<=5 else 0)
                candidate=(states+[state],actions+[action],new_good,total_ttft+ttft)
                next_paths.append(candidate)
        # Future arrivals and output lengths are permitted only in this oracle.
        # Prefix ranking can discard the best final path. No optimality claim.
        next_paths.sort(key=lambda p:(p[2]/max(p[0][-1]['D_finish'],1e-9),-p[3],-p[0][-1]['D_finish']),reverse=True)
        paths=next_paths[:width]
    return max(paths,key=lambda p:(metrics(p[0],arrivals,outputs)['modeled_TTFT_qualified_tokens_per_second'],-p[3]))


def conditional_validation(model,step_coefficients,patterns=('ControlledDiscovery_*_N200_q2.35_s2026112061_BI0_v*',)):
    results=[]
    for p in sorted({folder for pattern in patterns for folder in (ROOT/'raw').glob(pattern)}):
        if not (p/'requests.jsonl').exists() or not (p/'window.json').exists():continue
        cfg=json.loads((p/'configuration.json').read_text());rows=sorted(records(p/'requests.jsonl'),key=lambda r:r['sample_id'])
        if len(rows)!=200:continue
        origin=json.loads((p/'window.json').read_text())['start'];action=cfg['method']+str(cfg.get('axis_cut',0))
        arrivals=[r['arrival']-origin for r in rows];costs=[]
        for row in rows:
            step=predict(step_coefficients,row['input_tokens'])/1000
            costs.append(action_costs(model,row,row['output_tokens'],step))
        states=fixed_trace(arrivals,costs,[action]*len(rows))
        actual=[r['TTFT_ms'] for r in rows];predicted=[1000*(s['D_first']-a) for s,a in zip(states,arrivals)]
        results.append(dict(name=p.name,conditional_on_observed_output_lengths=True,
            TTFT_error=error_summary(actual,predicted),
            predicted_TTFT_p95_ms=float(np.percentile(predicted,95)),actual_TTFT_p95_ms=float(np.percentile(actual,95)),
            predicted_duration=max(s['D_finish'] for s in states),actual_duration=max(r['finish'] for r in rows)-origin))
    return results


def main():
    from .service_model import main as fit_models,fit
    fit_models();model=json.loads((ROOT/'analysis/BI0_serial_service_model_v2.json').read_text())
    native=json.loads((ROOT/'analysis/ContinuationCost_Native0_calibration20_BI0_v1_axis_timeline.json').read_text())['rows']
    usable=[r for r in native if r['output_tokens']>1]
    step_coefficients=fit([r['input_tokens'] for r in usable],[r['worker_first_to_last_output_ms']/(r['output_tokens']-1) for r in usable])
    validation=conditional_validation(model,step_coefficients)
    write(ROOT/'analysis/BI0_tandem_predictive_validation_v1.json',dict(results=validation,
        evidence='OFFLINE_REPLAY',scope='Service models use calibration20 only. Output lengths are the observed future values. '
        'This tests conditional service/queue prediction, not natural EOS prediction or online observability. '
        'Client ITL tails are not modeled. A TTFT-qualified token proxy is not full joint-SLO goodput.'))
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    natural=json.loads((ROOT/'raw/BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1/natural_reference.json').read_text())['rows']
    output_map={r['prompt_id']:r['output_tokens'] for r in natural};outputs=[output_map[r['prompt_id']] for r in data]
    allowed=list(model['actions']);cells=[]
    for rate in (1.2,1.65,2.35):
        frozen_fixed=None
        for seed in (2026112061,2026112062,2026112063):
            arrivals=np.cumsum(np.random.default_rng(seed).exponential(1/rate,len(data)));arrivals-=arrivals[0]
            costs=[action_costs(model,r,o,predict(step_coefficients,r['input_tokens'])/1000) for r,o in zip(data,outputs)]
            fixed=[]
            for action in allowed:
                for fraction in ((1.,) if action=='Native0' else (.25,.5,.75,1.)):
                    actions=[action if int(hashlib.sha256(r['prompt_id'].encode()).hexdigest()[:8],16)/2**32<fraction else 'Native0' for r in data]
                    states=fixed_trace(arrivals,costs,actions)
                    migrated=sum(value!='Native0' for value in actions)
                    fixed.append(dict(action=action,fraction=fraction,actual_migrated_requests=migrated,
                        actual_migrated_fraction=migrated/len(actions),
                        fraction_scope='Configured hash threshold, not an exact request fraction.',
                        metrics=metrics(states,arrivals,outputs),actions=actions))
            best=max(fixed,key=lambda r:r['metrics']['modeled_TTFT_qualified_tokens_per_second'])
            if frozen_fixed is None:frozen_fixed=(best['action'],best['fraction'])
            frozen_result=next(r for r in fixed if (r['action'],r['fraction'])==frozen_fixed)
            oracles={}
            for axis in ('Layer','Token','All'):
                actions=[a for a in allowed if a=='Native0' or axis=='All' or a.startswith(axis)]
                state,selected,_,_=beam(arrivals,costs,outputs,actions)
                candidate=dict(metrics=metrics(state,arrivals,outputs),actions=selected,states=state)
                fixed_axis=max((r for r in fixed if r['action'] in actions),key=lambda r:r['metrics']['modeled_TTFT_qualified_tokens_per_second'])
                if fixed_axis['metrics']['modeled_TTFT_qualified_tokens_per_second']>candidate['metrics']['modeled_TTFT_qualified_tokens_per_second']:
                    candidate=dict(metrics=fixed_axis['metrics'],actions=fixed_axis['actions'],states=fixed_trace(arrivals,costs,fixed_axis['actions']),fixed_fallback=True)
                oracles[axis]=candidate
            denominator=max(oracles[a]['metrics']['modeled_TTFT_qualified_tokens_per_second'] for a in ('Layer','Token'))
            cells.append(dict(QPS=rate,seed=seed,best_fixed_per_cell_hindsight=best,
                discovery_frozen_fixed=frozen_result,discovery_seed=2026112061,
                fixed_candidates=fixed,oracles=oracles,
                axis_additional_proxy_headroom=oracles['All']['metrics']['modeled_TTFT_qualified_tokens_per_second']/denominator-1 if denominator else None))
    write(ROOT/'analysis/BI0_conditional_axis_oracle_v1.json',dict(cells=cells,validation=validation,
        service_model_path='analysis/BI0_serial_service_model_v2.json',
        service_model_SHA256=digest(ROOT/'analysis/BI0_serial_service_model_v2.json'),
        decode_step_coefficients_ms=step_coefficients,evidence='OFFLINE_ORACLE',beam_width=24,
        online_axis_gate='HOLD_NOT_FULL_SLO_VALIDATED',
        scope='FIFO P1/D1/source-credit4 state updates. Fixed C1 output lengths preserve modeled output work. '
        'Oracle sees future arrivals/EOS. Finite beam is not a global bound. No client ITL-tail model, accuracy model, '
        'continuous batching, active-decode insertion or measured serving speedup. The proxy cannot by itself pass the 5% online-axis gate.'))
    audit=[dict(QPS=cell['QPS'],seed=cell['seed'],axis=axis,
        accounting=KV_contract_accounting(value['states'],data,outputs))
        for cell in cells for axis,value in cell['oracles'].items()]
    write(ROOT/'analysis/BI0_conditional_KV_credit_audit_v1.json',dict(results=audit,
        evidence='STRUCTURAL_ACCOUNTING',scope='Action-dependent state trajectories are checked. '
        'The conservative KV constraint is nonbinding under the tested fixed slot contract. No enlarged source-credit contract is certified.'))


if __name__=='__main__':main()

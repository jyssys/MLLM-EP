"""Test two visible-state scalars in a conditional queue-updating replay."""
import collections
import json
import math
import time
import numpy as np
from .common import ROOT,write
from .service_model import predict
from .offline_headroom import action_costs,extend,metrics


def choose_action(costs,D_work,backlog,reference_job,horizon):
    """Use estimated D work and visible P backlog. Do not read future EOS."""
    n=min(max(0,int(backlog)),horizon);scores={}
    for action,(P,F,D) in costs.items():
        first=max(P,D_work)+F;P_clock=P;D_clock=first+D;score=first
        for _ in range(n):
            p,f,d=reference_job;P_clock+=p;first=max(P_clock,D_clock)+f
            D_clock=first+d;score+=first
        scores[action]=score
    return min(scores,key=lambda action:(scores[action],action)),scores


def expected_remaining(samples,emitted):
    eligible=[value-emitted for value in samples if value>emitted]
    # This is a prior estimate, not a deadline bound. No current future EOS.
    return float(np.mean(eligible)) if eligible else max(1.,float(np.mean(samples))/2)


def same_selection(frozen,selected,scores,prior):
    canonical={str(key):value for key,value in scores.items()}
    return frozen['horizon']==selected and frozen['discovery_scores']==canonical and frozen['task_output_prior']==prior


def simulate(arrivals,rows,actual_costs,model,step_coefficients,task_samples,reference_job,horizon,state_mode='BOTH'):
    assert state_mode in ('BOTH','D_ONLY','METADATA_ONLY')
    states=[];actions=[];observations=[]
    for index,(arrival,row) in enumerate(zip(arrivals,rows)):
        release=states[index-4]['D_first'] if index>=4 else 0.
        now=max(arrival,states[-1]['P_finish'] if states else 0.,release)
        waiting=int(sum(a<=now for a in arrivals[index+1:]))
        D_work=0.
        for old,state in enumerate(states):
            # The simulator determines which completion events already occurred.
            # Future finish/EOS values are not exported to the decision function.
            if state['D_finish']<=now:continue
            previous=rows[old];step=predict(step_coefficients,previous['input_tokens'])/1000
            prior=task_samples[previous['task']]
            predicted=action_costs(model,previous,float(np.mean(prior)),step)[actions[old]]
            if state['D_start']>now:D_work+=predicted[1]+predicted[2]
            elif state['D_first']>now:
                D_work+=max(0.,predicted[1]-(now-state['D_start']))+predicted[2]
            else:
                emitted=1+int(math.floor(max(0.,now-state['D_first'])/step))
                D_work+=expected_remaining(prior,emitted)*step
        step=predict(step_coefficients,row['input_tokens'])/1000
        prior=float(np.mean(task_samples[row['task']]))
        expected=action_costs(model,row,prior,step)
        used_D=0. if state_mode=='METADATA_ONLY' else D_work
        used_P=waiting if state_mode=='BOTH' else 0
        action,scores=choose_action(expected,used_D,used_P,reference_job,horizon)
        state=extend(states,index,arrival,actual_costs[index][action])
        states.append(state);actions.append(action)
        observations.append(dict(index=index,decision_time=now,P_visible_backlog=waiting,
            D_estimated_remaining_work_seconds=D_work,state_mode=state_mode,
            decision_D_work_seconds=used_D,decision_P_backlog=used_P,action=action,scores=scores,
            scope='Only past completions/emissions and calibration priors inform the choice.'))
    return states,actions,observations


def main():
    design=ROOT/'configs/frozen_two_state_rule_design_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),online_state_scalars=['P_visible_backlog','D_estimated_remaining_work_seconds'],
            action_metadata=['Current input length','Task-conditioned calibration output-length prior','Calibrated action service costs'],
            horizon_candidates=[0,1,2,4],discovery_seed=2026112061,heldout_seeds=[2026112062,2026112063],
            selection='Highest sum of TTFT-qualified token proxy across discovery QPS1.2/1.65/2.35. Freeze before held-out replay.',
            scope='Offline conditional replay only. Actual future arrivals/EOS drive the hidden simulator, never the decision API. '
            'The rule lacks physical safe budgets, ITL tails and task-quality dynamics. It cannot authorize an online axis controller.'))
    while not (ROOT/'state/offline_conditional_replay_complete.json').exists():
        for marker in ('STOP_REQUESTED.json','replay_campaign_hold.json','continuation_costs_hold.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(5)
    oracle=json.loads((ROOT/'analysis/BI0_conditional_axis_oracle_v1.json').read_text())
    model=json.loads((ROOT/'analysis/BI0_serial_service_model_v2.json').read_text())
    steps=oracle['decode_step_coefficients_ms']
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    natural=json.loads((ROOT/'raw/BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1/natural_reference.json').read_text())['rows']
    output_map={r['prompt_id']:r['output_tokens'] for r in natural};outputs=[output_map[r['prompt_id']] for r in data]
    calibration=json.loads((ROOT/'analysis/ContinuationCost_Native0_calibration20_BI0_v1_axis_timeline.json').read_text())['rows']
    samples=collections.defaultdict(list)
    for row in calibration:samples[row['task']].append(row['output_tokens'])
    prior={task:list(values) for task,values in samples.items()}
    results=[];selected=None
    for seed in (2026112061,2026112062,2026112063):
        for rate in (1.2,1.65,2.35):
            cell=next(r for r in oracle['cells'] if r['seed']==seed and r['QPS']==rate)
            default=cell['discovery_frozen_fixed']['action']
            virtual=[action_costs(model,row,float(np.mean(prior[row['task']])),predict(steps,row['input_tokens'])/1000)[default] for row in calibration]
            reference_job=np.mean(virtual,axis=0).tolist()
            arrivals=np.cumsum(np.random.default_rng(seed).exponential(1/rate,len(data)));arrivals-=arrivals[0]
            actual=[action_costs(model,row,o,predict(steps,row['input_tokens'])/1000) for row,o in zip(data,outputs)]
            horizons=(0,1,2,4) if seed==2026112061 else (selected,)
            for horizon in horizons:
                state,actions,observations=simulate(arrivals,data,actual,model,steps,prior,reference_job,horizon)
                result=metrics(state,arrivals,outputs);fixed=cell['discovery_frozen_fixed']['metrics']
                denominator=fixed['modeled_TTFT_qualified_tokens_per_second']
                results.append(dict(QPS=rate,seed=seed,horizon=horizon,metrics=result,
                    over_discovery_frozen_fixed_proxy=result['modeled_TTFT_qualified_tokens_per_second']/denominator if denominator else None,
                    selected_actions=dict(collections.Counter(actions)),decisions=observations))
        if seed==2026112061:
            scores={h:sum(r['metrics']['modeled_TTFT_qualified_tokens_per_second'] for r in results if r['horizon']==h) for h in (0,1,2,4)}
            selected=max(scores,key=lambda h:(scores[h],-h))
            selection=ROOT/'configs/frozen_two_state_rule_selection_v1.json'
            if selection.exists():
                frozen=json.loads(selection.read_text())
                assert same_selection(frozen,selected,scores,prior)
            else:
                write(selection,dict(time=time.time(),horizon=selected,
                    discovery_scores=scores,task_output_prior=prior,scope='Selected before held-out seeds62/63. No online GPU policy is enabled.'))
    write(ROOT/'analysis/BI0_two_state_rule_replay_v1.json',dict(results=results,selected_horizon=selected,
        online_state_scalars=['P backlog count','Estimated remaining D work'],task_prior=prior,
        evidence='OFFLINE_REPLAY',online_gate='HOLD_NOT_FULL_SLO_VALIDATED',
        scope='Queue and source-credit states update after each action. Decisions exclude future arrivals and current-request future EOS. '
        'Input length/task and calibrated action costs are metadata, not omitted state variables. The hidden simulator uses fixed future output lengths. '
        'No client ITL-tail, physical-window or accuracy model. No measured online speedup or proven two-knob generalization.'))
    ablations=[]
    for seed in (2026112061,2026112062,2026112063):
        for rate in (1.2,1.65,2.35):
            cell=next(r for r in oracle['cells'] if r['seed']==seed and r['QPS']==rate)
            default=cell['discovery_frozen_fixed']['action']
            virtual=[action_costs(model,row,float(np.mean(prior[row['task']])),predict(steps,row['input_tokens'])/1000)[default] for row in calibration]
            reference_job=np.mean(virtual,axis=0).tolist()
            arrivals=np.cumsum(np.random.default_rng(seed).exponential(1/rate,len(data)));arrivals-=arrivals[0]
            actual=[action_costs(model,row,o,predict(steps,row['input_tokens'])/1000) for row,o in zip(data,outputs)]
            for mode in ('METADATA_ONLY','D_ONLY'):
                state,actions,observations=simulate(arrivals,data,actual,model,steps,prior,reference_job,0,state_mode=mode)
                ablations.append(dict(seed=seed,QPS=rate,state_mode=mode,metrics=metrics(state,arrivals,outputs),
                    selected_actions=dict(collections.Counter(actions)),decisions=observations))
            full=next(r for r in results if r['seed']==seed and r['QPS']==rate and r['horizon']==selected)
            ablations.append(dict(seed=seed,QPS=rate,state_mode='BOTH_DISCOVERY_FROZEN_HORIZON',
                horizon=selected,metrics=full['metrics'],selected_actions=full['selected_actions']))
    write(ROOT/'analysis/BI0_visible_state_rule_ablation_v1.json',dict(results=ablations,
        evidence='OFFLINE_REPLAY',scope='Metadata-only and D-state-only ablations were declared before fitting the service model. '
        'All use the same calibrated costs, task priors, output work and updating queues. '
        'These conditional proxy results do not establish online observability of physical safe budgets or full SLO gains.'))
    write(ROOT/'state/two_state_rule_replay_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/two_state_rule_replay_hold.json',dict(time=time.time(),error=repr(error)));raise

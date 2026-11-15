"""Replay frozen online decisions without future request information."""
import argparse,collections,copy
from dataclasses import asdict,replace
from .common import *
from .analyze import records,percentile
from .policy import LayerPolicy,Observation


def replay(name,changes=None,independent_gates=False):
    cfg=json.loads((PREVIOUS/'configs'/(name+'.json')).read_text())
    cfg.update(changes or {})
    costs=json.loads((PREVIOUS/'configs'/cfg['cost_model']).read_text())
    policy=LayerPolicy(cfg,costs);selected=collections.Counter();reasons=collections.Counter()
    gates={str(m):collections.Counter() for m in (2,4,8,12)};mismatches=[];details=[]
    for index,row in enumerate(records(PREVIOUS/'raw'/name/'online_decisions.jsonl')):
        state=Observation(**row['observation']);choice=policy.choose(state)
        selected[choice.layers]+=state.group_size;reasons[choice.reason]+=state.group_size
        if not changes and asdict(choice)!=row['choice']:mismatches.append(index)
        global_gates=[]
        if state.state_age_ms>cfg.get('maximum_state_age_ms',250):global_gates.append('STALE_D_STATE')
        if state.P_backlog<cfg.get('minimum_P_backlog',1):global_gates.append('NO_P_COMPUTE_BACKLOG')
        blocks=state.group_size*__import__('math').ceil((cfg['length']+cfg['cap'])/16)
        if state.KV_free_blocks<blocks:global_gates.append('KV_CREDIT_BOUND')
        # Expose all predicates even when the production policy returns early.
        # These costs do not turn missing backlog or missing credit into work.
        probe=LayerPolicy(cfg,costs)
        probe.seen_requests=policy.seen_requests;probe.selected_requests=policy.selected_requests
        probe.selected_layer_work=policy.selected_layer_work
        candidates=probe.revised(replace(state,state_age_ms=0.,
            P_backlog=max(state.P_backlog,cfg.get('minimum_P_backlog',1)),
            KV_free_blocks=max(blocks,state.KV_free_blocks))).candidates
        for candidate in candidates:
            labels=set(global_gates+candidate['rejection_reasons'])
            for reason in labels:gates[str(candidate['layers'])][reason]+=state.group_size
            if not labels:gates[str(candidate['layers'])]['ALL_PREDICATES_PASS']+=state.group_size
        details.append(dict(decision_index=index,time=row['time'],requests=state.group_size,
            global_rejections=global_gates,observation=asdict(state),
            production_choice=asdict(choice),independent_candidates=candidates))
    return dict(name=name,changes=changes or {},selected_requests=dict(selected),
        choice_reasons=dict(reasons),candidate_overlapping_gate_request_counts=gates,
        replay_mismatch_indices=mismatches,decisions=len(details),details=details,
        evidence='OFFLINE_ORACLE' if changes else 'TRACE_ANALYSIS',
        scope='Replay uses actual observed Host state. Counterfactual choices do not update subsequent real states. Gate counts overlap and must not be summed as distinct requests. This does not measure GPU headroom or online migration.')


def run(require_migration=False):
    initialize();summary=[]
    for cap in (64,200):
        for rate in ((2.5,4.8,6.4) if cap==64 else (1.5,2.5,4.8)):
            for seed in (2026111381,2026111382,2026111383):
                name=f'heldout_Revised_O{cap}_N128_q{rate}_s{seed}_v1'
                if not (PREVIOUS/'configs'/(name+'.json')).exists():continue
                result=replay(name);assert not result['replay_mismatch_indices'],name
                write(ROOT/'analysis'/(name+'_gate_replay.json'),result)
                variations={}
                for label,change in [('wait_50',dict(maximum_predicted_D_wait_ms=50.)),
                    ('wait_250',dict(maximum_predicted_D_wait_ms=250.)),
                    ('wait_1000',dict(maximum_predicted_D_wait_ms=1000.)),
                    ('drain_margin_zero',None),('safety_margin_zero',dict(safety_margin_by_layers_ms={str(m):0. for m in (2,4,8,12)})),
                    ('without_backlog_gate',dict(minimum_P_backlog=0))]:
                    if change is None:continue
                    value=replay(name,change)
                    variations[label]={k:v for k,v in value.items() if k!='details'}
                summary.append({**{k:v for k,v in result.items() if k!='details'},'counterfactuals':variations})
    write(ROOT/'analysis/long_output_gate_attribution.json',dict(runs=summary,
        evidence='TRACE_ANALYSIS',overlapping_predicates=True,
        source='Frozen primary online_decisions.jsonl and actual LayerPolicy code at publication parent.'))
    if require_migration:
        assert any(sum(v for k,v in row['selected_requests'].items() if int(k)>0) for row in summary), 'REPRODUCED: all frozen long-output choices are Native'
    print(json.dumps(dict(runs=len(summary),exact_replay=True,
        selected=[r['selected_requests'] for r in summary]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-migration',action='store_true');a=p.parse_args();run(a.require_migration)

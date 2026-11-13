"""Calibrate residual bounds after correcting serialized handoff prediction."""
import math
from .common import *
from .policy import LayerPolicy,Observation
from .analyze import records,percentile


def main():
    target=ROOT/'configs/revised_cost_model_v3.json'
    if target.exists():print('V3_MODEL_ALREADY_FROZEN');return
    assert not (ROOT/'configs/heldout_selection.json').exists(),'Do not tune after fresh evaluation selection'
    model=json.loads((ROOT/'configs/revised_cost_model_v2.json').read_text())
    folders=list((ROOT/'raw').iterdir())+[PREVIOUS/'raw'/f'heldout_O16_N128_s{s}_COST_v1' for s in range(2026111341,2026111346)]
    samples=[];sources=[]
    for folder in folders:
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
        if not (folder/'online_decisions.jsonl').exists():continue
        cfg=json.loads((folder/'configuration.json').read_text())
        if cfg.get('cap')!=16 or cfg.get('policy') not in ('REVISED','COST','STATIC'):continue
        if any(cfg.get(k) for k in ('execution_events','verify_layer_execution','full_prompt_state_witness',
            'numeric_probe','correctness_only','completion_handler_timing','state_capture')):continue
        requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
        commits={r['generation']:r for r in records(folder/'actual_cut_commits.jsonl')}
        cfg.update(policy='REVISED',revised_revision=2,safety_margin_ms=0.,
            maximum_predicted_D_wait_ms=250.,maximum_queue_groups=8)
        cfg.pop('safety_margin_by_layers_ms',None);policy=LayerPolicy(cfg,model);used=0
        for decision in records(folder/'online_decisions.jsonl'):
            actual=commits.get(decision['generation'])
            if not actual or not actual['offloaded_layers'] or not all(r in requests for r in actual['requests']):continue
            choice=policy.choose(Observation(**decision['observation']))
            row=next((c for c in choice.candidates if c['layers']==actual['offloaded_layers']),None)
            if row is None or not row['accepted']:continue
            error=1000*(max(requests[r]['first_token'] for r in actual['requests'])-row['predicted_D_finish'])
            samples.append(dict(name=folder.name,layers=actual['offloaded_layers'],batch=len(actual['requests']),error_ms=error));used+=1
        if used:sources.append(dict(name=folder.name,samples=used,
            decisions_SHA256=digest(folder/'online_decisions.jsonl'),requests_SHA256=digest(folder/'requests.jsonl')))
    margins={};summary={}
    for layers in (2,4,8,12):
        values=[r['error_ms'] for r in samples if r['layers']==layers]
        runs={r['name'] for r in samples if r['layers']==layers}
        # Retain110ms without enough observed support. Otherwise cover every
        # retained training residual, with at least50ms of margin.
        margins[str(layers)]=(max(50.,10*math.ceil(max(0.,max(values))/10))
            if len(values)>=8 and len(runs)>=2 else 110.)
        summary[str(layers)]=dict(samples=len(values),viewed_starts=len(runs),
            error_p95_ms=percentile(values,.95),maximum_error_ms=max(values) if values else None,
            selected_margin_ms=margins[str(layers)],
            retrospective_coverage=sum(e<=margins[str(layers)] for e in values)/len(values) if values else None)
    write(ROOT/'analysis/v3_uncertainty_training.json',dict(samples=samples,sources=sources,summary=summary,
        evidence='TRACE_ANALYSIS',fresh_confirmation_information=False,
        scope='Retrospective replay on previously viewed clean Output16 runs. Rows can be correlated. Training coverage is not validation, a p99 confidence bound, or WCET.'))
    model.update(v3_freeze_epoch=time.time(),v2_SHA256=digest(ROOT/'configs/revised_cost_model_v2.json'),
        Output16_safety_margin_by_layers_ms=margins,
        residual_training_SHA256=digest(ROOT/'analysis/v3_uncertainty_training.json'))
    write(target,model);print('V3_FROZEN_MARGINS',margins,summary,flush=True)


if __name__=='__main__':main()

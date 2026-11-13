"""Evaluate predeclared ablations and aggregate-profile matched controls."""
import random
from .common import *
from .configuration import base
from .run import execute
from .analyze import records,compare,metrics


def controls(dynamic_name,selection):
    original=json.loads((ROOT/'configs'/(dynamic_name+'.json')).read_text())
    rows=records(ROOT/'raw'/dynamic_name/'actual_cut_commits.jsonl')
    # Retain the measured group-size/cut multiset. Assign it to different
    # request indices without using any per-request latency or prompt value.
    blocks=[(len(r['requests']),r['offloaded_layers']) for r in rows]
    rng=random.Random(original['arrival_seed']^0x35A117);rng.shuffle(blocks)
    assignment=[layers for count,layers in blocks for _ in range(count)]
    assert len(assignment)==original['count']
    kwargs=dict(count=128,cap=16,rate=4.8,seed=original['arrival_seed'],dataset='fresh_confirmation.json',
        phase='PREDECLARED_SECONDARY_CONTROL_ON_FRESH_PROMPTS')
    common=dict(selection['settings_by_cap']['16']['Revised'])
    static_keys={k:common[k] for k in ('D_admission_batch','D_token_budget','max_reserved_requests',
        'coordinator_GIL_switch_interval_s','defer_completion_log','native_continuous_D_ingress') if k in common}
    jobs=[]
    cfg=base(f'heldout_secondary_ControllerZero_s{original["arrival_seed"]}_v1',arm='NATIVE',
        **kwargs,**static_keys)
    jobs.append(('ControllerZero',cfg))
    for fraction in (.3125,.5):
        cfg=base(f'heldout_secondary_Static12_f{fraction:g}_s{original["arrival_seed"]}_v1',
            arm='STATIC',static_layers=12,migration_fraction=fraction,
            static_fraction_axis='REQUEST_ID_BLOCKS',**kwargs,**static_keys)
        jobs.append(('Static3125' if fraction==.3125 else 'Static50',cfg))
    cfg=base(f'heldout_secondary_MatchedProfile_s{original["arrival_seed"]}_v1',
        arm='STATIC',static_layers=12,static_layers_by_request=assignment,**kwargs,**static_keys)
    cfg['matched_control_scope']='Post-hoc aggregate-profile control. Only the completed group-size/cut multiset determines its budget. Request assignment uses a fixed independent shuffle. It is not a pre-selected deployable policy or a new held-out winner.'
    cfg['matching_source']=dynamic_name
    cfg['matched_expected_requests']=sum(x>0 for x in assignment)
    cfg['matched_expected_layers']=sum(assignment)
    jobs.append(('MatchedProfile',cfg))
    for arm in ('IDLE','DEADLINE','COST'):
        cfg=base(f'heldout_secondary_{arm}_s{original["arrival_seed"]}_v1',arm=arm,
            safety_margin_ms=5.,**kwargs,**static_keys)
        jobs.append((arm,cfg))
    return jobs


def execute_controls(dynamic_name,selection,reference):
    state_path=ROOT/'state/secondary_progress.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(results=[],not_run=[])
    present={r['name'] for r in state['results']}
    for arm,cfg in controls(dynamic_name,selection):
        name=save(cfg)
        if name in present:continue
        if time.time()>=selection['experiment_deadline_epoch']:
            state['not_run'].append(dict(name=name,status='NOT_RUN',reason='CAMPAIGN_EXPERIMENT_DEADLINE'));continue
        if not execute(name,restore=False):state['failed']=name;write(state_path,state);return False
        output=compare(reference,name,True);requests=records(ROOT/'raw'/name/'requests.jsonl')
        migrated=sum(r.get('completed_D_prefill_layers',0)>0 for r in requests)
        layers=sum(r.get('completed_D_prefill_layers',0) for r in requests)
        matched=(migrated==cfg['matched_expected_requests'] and layers==cfg['matched_expected_layers']) if arm=='MatchedProfile' else None
        state['results'].append(dict(name=name,arm=arm,cap=16,rate=4.8,seed=cfg['arrival_seed'],
            output=output,metrics=metrics(name),migrated_requests=migrated,migrated_layers=layers,
            matched_scope=cfg.get('matched_control_scope'),realized_budget_match=matched,
            expected_matched_requests=cfg.get('matched_expected_requests'),expected_matched_layers=cfg.get('matched_expected_layers')))
        write(state_path,state);print('SECONDARY',name,output['passed'],migrated,layers,flush=True)
        if not output['passed']:return False
    write(state_path,state);return True

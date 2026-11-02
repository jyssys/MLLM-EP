"""Stop further model expansion after an unexplained measured output difference."""
import hashlib
import importlib
from .common import *
from .evidence import rows,status,full_work,atomic
from . import families
importlib.reload(families)
from .families import execution_family

PERFORMANCE_PHASES={'qps_pilot','final_repeat','profile_control','paper_budget_control'}


def blocker_key(cfg):
    if cfg.get('ar_execution_family'):return cfg['model']+'|'+cfg['ar_execution_family']+'|L'+str(cfg['length'])
    if cfg.get('execution_control_family'):return cfg['model']+'|L'+str(cfg['length'])+'|'+cfg['execution_control_family']
    if cfg.get('extension'):return cfg['model']+'|L'+str(cfg['length'])+'|'+execution_family(cfg)
    if cfg['arm'].startswith('DP'):
        return '|'.join(map(str,(cfg['model'],cfg['arm'],cfg['length'],cfg['cap'],cfg['chunk'],
            cfg['token_budget'],cfg['kv_blocks'],cfg.get('wavefront',False))))
    return cfg['model']+'|L'+str(cfg['length'])


def blocked(model,cfg=None):
    p=ROOT/'state/online_correctness_blockers.json'
    values=json.loads(p.read_text()) if p.exists() else {}
    if cfg is not None and cfg.get('ar_execution_family'):return values.get(blocker_key(cfg))
    legacy='|'.join(map(str,(cfg['model'],cfg['arm'],cfg['length'],cfg['cap'],cfg['chunk']))) if cfg is not None and cfg['arm'].startswith('DP') else None
    return values.get(model) or values.get(legacy) or (values.get(blocker_key(cfg)) if cfg is not None else None)


def verify(cfg):
    from . import profile_prerequisite_gate
    import importlib
    importlib.reload(profile_prerequisite_gate)
    try:profile_prerequisite_gate.advance()
    except Exception as error:
        import traceback
        append(ROOT/'state/profile_prerequisite_gate_errors.jsonl',dict(time=time.time(),error=repr(error),traceback=traceback.format_exc(),
            scope='CPU gate error. Preserve evidence and continue required output/work audits. Do not infer a performance pass.'))
    from .h_workspace_controls import advance
    try:advance()
    except Exception as error:
        import traceback
        append(ROOT/'state/control_generation_errors.jsonl',dict(time=time.time(),error=repr(error),traceback=traceback.format_exc(),
            scope='Optional control generation failed. Do not claim its gate passed. Existing qualified jobs can continue.'))
    if cfg.get('phase') not in PERFORMANCE_PHASES:return
    if status(cfg['name'])=='FAILED' and not any(cfg.get(k) for k in ('nsys','nsys_P_only')):
        dispatcher=ROOT/'raw'/cfg['name']/'dispatcher_result.json'
        if dispatcher.exists() and json.loads(dispatcher.read_text()).get('stop_reason')=='USER_11H_DEADLINE':
            append(ROOT/'state/deadline_truncated_runs.jsonl',dict(time=time.time(),name=cfg['name'],
                scope='Budget truncation. Do not invalidate earlier complete runs or diagnose a numerical failure from this cancellation.'))
            return
        path=ROOT/'raw'/cfg['name']/'status.json';detail=json.loads(path.read_text()) if path.exists() else {}
        p=ROOT/'state/online_correctness_blockers.json';blocks=json.loads(p.read_text()) if p.exists() else {}
        blocks[blocker_key(cfg)]=dict(time=time.time(),name=cfg['name'],classification='RUNTIME_FAILURE_REQUIRES_TRIAGE',
            error=detail.get('error','No valid terminal status'),scope='Stop expansion until diagnosis. This does not establish a numerical bug.')
        write(p,blocks);return
    if status(cfg['name'])!='COMPLETE':return
    path=ROOT/'state/canonical_outputs.json';canonical=json.loads(path.read_text()) if path.exists() else {}
    failures=[];compared=0;new=0
    for r in rows(cfg['name']):
        input_sha=hashlib.sha256(json.dumps(r['input_ids'],separators=(',',':')).encode()).hexdigest()
        family=('AR_COLOCATED' if cfg['arm']=='DP2' else 'AR_NATIVE_PD') if cfg['model']=='qwen8' else 'QUALIFIED_RLT'
        if cfg.get('ar_execution_family'):family='AR_RECOVERY_'+cfg['ar_execution_family']
        context_geometry=cfg['length']+cfg['cap'] if cfg['model']=='qwen8' else cfg['context_limit']
        key='|'.join(map(str,(cfg['model'],cfg['model_revision'],cfg.get('ar_batch_invariant',False),family,context_geometry,r['seed'],input_sha)))
        previous=canonical.get(key)
        if previous:
            compared+=1;n=min(len(previous['output_ids']),len(r['output_ids']))
            difference=next((i for i in range(n) if previous['output_ids'][i]!=r['output_ids'][i]),None)
            if difference is not None:failures.append(dict(sample_id=r['sample_id'],reference=previous['name'],
                candidate=cfg['name'],first_divergent_token=difference,reference_ids=previous['output_ids'],candidate_ids=r['output_ids']))
        if previous is None or (not failures and len(r['output_ids'])>len(previous['output_ids'])):
            canonical[key]=dict(name=cfg['name'],output_ids=r['output_ids']);new+=1
    work=full_work(cfg['name']) if cfg['model']!='qwen8' else dict(passed=True,scope='Native AR, no recurrent work counter.')
    span=atomic(cfg['name']) if cfg['atomic_prompt'] and cfg['arm'] not in ('DP','DP2','DP_Chunked') else dict(passed=True,scope='No Atomic-P assertion for chunked or colocated controls.')
    output_scope=('The named AR recovery family shares canonical output checks across native PD and DP2. '
        if cfg.get('ar_execution_family') else
        'Original AR Native-PD and colocated families retain separate canonical outputs because their cross-mode gate failed. ')
    result=dict(time=time.time(),name=cfg['name'],previously_seen_inputs=compared,new_or_extended_references=new,
        output_failures=failures,work=work,atomic=span,passed=not failures and work['passed'] and span['passed'],
        scope='Exact observed common output prefix for matched input IDs, seed, and native max-context geometry. '+output_scope+
            'Within-cell C, rate, and independent-start comparisons remain exact. Not original-HF fidelity or downstream accuracy.')
    append(ROOT/'state/live_correctness_audits.jsonl',result)
    if not result['passed']:
        p=ROOT/'state/online_correctness_blockers.json';blocks=json.loads(p.read_text()) if p.exists() else {}
        blocks[blocker_key(cfg)]=result;write(p,blocks)
    else:write(path,canonical)

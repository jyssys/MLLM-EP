"""Reuse metric definitions with explicit process-local evidence routing."""
from .common import *
from ..idle_intervention_dynamic_loop.analyze import percentile


def qualification_config_view(value,name):
    if (isinstance(value,dict) and value.get('name')==name and value.get('native_path')
            and value.get('feature_native_wire_v5') and value.get('native_partial_cut') is None):
        # Native-path install resolves a null partial cut to the full depth.
        # This analysis view does not change the immutable runtime config.
        return dict(value,native_partial_cut=value['depth'])
    return value


def metrics(name):
    configure()
    from ..idle_intervention_dynamic_loop import analyze,qualification
    analyze.ROOT=ROOT;analyze.configure=configure
    qualification.ROOT=ROOT;qualification.configure=configure
    return analyze.metrics(name)


def qualify(name,reference=None):
    configure()
    from ..idle_intervention_dynamic_loop import qualification
    qualification.ROOT=ROOT;qualification.configure=configure
    previous=qualification.json
    class ConfigJSON:
        def loads(self,text,*args,**kwargs):
            return qualification_config_view(json.loads(text,*args,**kwargs),name)
        def __getattr__(self,key):return getattr(json,key)
    qualification.json=ConfigJSON()
    try:result=qualification.qualify(name,reference)
    finally:qualification.json=previous
    cfg_path=ROOT/'configs'/(name+'.json')
    cfg=json.loads(cfg_path.read_text())
    if qualification_config_view(cfg,name)!=cfg:
        result['analysis_configuration_view']=dict(original_sha256=digest(cfg_path),
            raw_native_partial_cut=None,effective_native_partial_cut=cfg['depth'],
            scope='Native-path runtime resolves null to full depth. This view fixes inherited audit arithmetic only. Output, KV, hidden, and lifetime gates remain unchanged.')
        write(ROOT/'raw'/name/'qualification.json',result)
    return result


def qualify_frozen_reference(name):
    result=qualify(name)
    if not result['complete']:return result
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    reference=OLD/'raw'/f"n32_{cfg['model']}_native_full_state32"
    expected=set(range(cfg['seed'],cfg['seed']+cfg['count']))
    def outputs(path):
        return {r['seed']:r for r in map(json.loads,(path/'requests.jsonl').read_text().splitlines())
            if not r.get('warmup') and r['seed'] in expected}
    def states(path):
        values={}
        for p in path.glob('prompt_states_*.jsonl'):
            for r in map(json.loads,p.read_text().splitlines()):
                if r['seed'] not in expected or r['prompt_tokens']!=cfg['length']:continue
                item=(r['hidden_sha256'],r['planes'])
                if r['seed'] in values:assert values[r['seed']]==item,'Conflicting state records'
                values[r['seed']]=item
        return values
    a,b=outputs(reference),outputs(folder);x,y=states(reference),states(folder)
    output_fail=[seed for seed in expected if seed not in a or seed not in b or
        any(a[seed][k]!=b[seed][k] for k in ('input_ids','output_ids','seed'))]
    state_fail=[seed for seed in expected if seed not in x or seed not in y or x[seed]!=y[seed]]
    result['frozen_output']=dict(reference=str(reference),expected=len(expected),failures=output_fail,passed=not output_fail)
    result['frozen_full_state']=dict(reference=str(reference),expected=len(expected),failures=state_fail,passed=not state_fail,
        scope='All recorded valid KV-plane hashes and prompt hidden hashes. Frozen qualified Native reference, not a new Native GPU run.')
    result['passed']=result['passed'] and not output_fail and not state_fail
    write(folder/'qualification_with_frozen_reference.json',result)
    return result

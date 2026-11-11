"""Qualify a new-length Native reference without inventing an old state hash."""
import json
from .common import *


def state_rows(folder,length):
    out={}
    for p in folder.glob('prompt_states_*.jsonl'):
        for r in map(json.loads,p.read_text().splitlines()):
            if length is not None and r['prompt_tokens']!=length:continue
            value=(r['hidden_sha256'],r['planes'])
            assert r['seed'] not in out or out[r['seed']]==value,'Conflicting prompt-state records'
            out[r['seed']]=value
    return out


def qualify(name):
    configure()
    from ..completion_aware_loop_pd import analyze
    analyze.ROOT=ROOT;analyze.configure=configure
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());reference=cfg.get('campaign_state_reference')
    result=analyze.qualify(name,reference)
    if not result.get('complete'):return result
    expected=set(range(cfg['seed'],cfg['seed']+cfg['count']))
    base=f'native_{cfg["model"]}_{"early_profile" if cfg["model"]=="ouro" else "clean_recovery"}_state8'
    base_cfg=json.loads((ROOT/'configs'/(base+'.json')).read_text())
    canonical=next(iter(state_rows(ROOT/'raw'/base,base_cfg['length']).values()))[1]
    layout={(r['depth'],r['layer'],r['kind']):base_cfg['length']-r['valid_tokens'] for r in canonical}
    lengths={seed:cfg['length'] for seed in expected}
    if cfg.get('prompt_token_lengths'):
        assert len(cfg['prompt_token_lengths'])==cfg['count']
        lengths=dict(zip(range(cfg['seed'],cfg['seed']+cfg['count']),cfg['prompt_token_lengths']))
    states=state_rows(ROOT/'raw'/name,None if cfg.get('prompt_token_lengths') else cfg['length']);bad=[]
    for seed in expected:
        if seed not in states:bad.append(seed);continue
        hidden,planes=states[seed]
        observed={(p['depth'],p['layer'],p['kind']):lengths[seed]-p['valid_tokens'] for p in planes}
        if observed!=layout or len(planes)!=len(layout) or len(hidden)!=64 or any(len(p['sha256'])!=64 for p in planes):bad.append(seed)
    result['new_length_state_coverage']=dict(expected=cfg['count'],valid_plane_hashes_per_request=len(layout),
        expected_prompt_lengths=sorted(set(lengths.values())),
        failures=bad,passed=not bad,
        scope='Full recorded state coverage uses the qualified model layout. New-length bytes are not compared with a different-length hash. A split must match its fresh same-length Native reference.')
    subset=cfg.get('qualification_state_subset_reference')
    if subset:
        previous_cfg=json.loads((ROOT/'configs'/(subset+'.json')).read_text())
        selected=set(range(previous_cfg['seed'],previous_cfg['seed']+previous_cfg['count']))
        assert selected<=expected and previous_cfg['length']==cfg['length']
        previous=state_rows(ROOT/'raw'/subset,previous_cfg['length'])
        different=[seed for seed in selected if seed not in previous or seed not in states or previous[seed]!=states[seed]]
        result['native_state_subset_comparison']=dict(reference=subset,expected=len(selected),failures=different,
            passed=not different,scope='Only the earlier N8 subset has an independent state hash. All new requests still require full state coverage and exact handoff. A subsequent split N32 must match this complete Native N32 reference.')
    output_ref=cfg.get('qualification_output_reference')
    if output_ref:
        def outputs(job):
            return {r['seed']:r for r in map(json.loads,(ROOT/'raw'/job/'requests.jsonl').read_text().splitlines())
                if not r.get('warmup') and r['seed'] in expected}
        a,b=outputs(output_ref),outputs(name)
        different=[seed for seed in expected if seed not in a or seed not in b or any(a[seed][k]!=b[seed][k] for k in ('input_ids','output_ids','seed'))]
        result['previous_native_output_subset']=dict(reference=output_ref,expected=cfg['count'],failures=different,
            passed=not different,scope='Only the requested seed subset enters this check. The larger reference has no prompt-state hashes for this geometry.')
    result['passed'] &= not bad and result.get('previous_native_output_subset',{}).get('passed',True) and result.get('native_state_subset_comparison',{}).get('passed',True)
    result['reference_scope']='FRESH_SAME_LENGTH_NATIVE_REFERENCE' if not reference else 'FRESH_SAME_LENGTH_NATIVE_STATE_COMPARISON'
    return result

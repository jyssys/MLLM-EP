"""Inspect the observed native plateau before independent final confirmation."""
from .common import *
from .analyze import serving
from .search import serving_config


def configurations():
    path=ROOT/'state/fine_QPS_plan.json'
    if path.exists():return []
    runs=[r for r in serving() if r['eligible_clean'] and r['phase']=='qps_pilot' and r['arm']=='PD_G1' and
        isinstance(r.get('P_completion_rate_central'),(int,float)) and isinstance(r.get('D_KV_refusal_sample_fraction'),(int,float))]
    references=json.loads((ROOT/'configs/capacity_references.json').read_text());new=[];choices={}
    for model in MODEL_PINS:
        pool=[r for r in runs if r['model']==model and r['D_TPOT_margin_pass']]
        if not pool:continue
        def score(r):
            cell=references[model+'_L'+str(r['length'])];muP=max(c['mu_P'] for c in cell['chunks'].values())
            muD=cell['D'][str(r['output_cap'])]['mu_D']
            return (int(muD>=muP),r['P_completion_rate_central']/muP,-r['D_KV_refusal_sample_fraction'],muD/muP)
        source=max(pool,key=score);cell=references[model+'_L'+str(source['length'])]
        baseline=json.loads((ROOT/'configs'/(source['name']+'.json')).read_text())
        center=min(max(r['mu_P'] for r in cell['chunks'].values()),source['P_completion_rate_central'])
        if center<=0:continue
        controls={k:baseline[k] for k in ('native_transport_limits','D_token_budget','D_kv_blocks') if k in baseline}
        choices[model]=dict(source=source['name'],native_completion_reference=center,
            rule='Explore 0.90/1.00/1.05/1.10 times an observed finite native completion rate. This is not a sustainable capacity estimate.')
        for chunk in dict.fromkeys([cell['length'],source['chunk'],cell['tuned_chunk']]):
            label='full' if chunk==cell['length'] else str(chunk)
            for factor in (.9,1.,1.05,1.1):
                c=serving_config(cell,f'fine_{model}_l{cell["length"]}_o{source["output_cap"]}_c{label}_f{str(factor).replace(".","p")}_v1',
                    source['output_cap'],chunk,'qps_pilot',64,rate=center*factor)
                c.update(controls);c.update(search_round='FINE_NATIVE_PLATEAU',reference_capacity=center,rate_factor=factor,
                    rate_scope='N64 exploratory screen. Independent final starts use different arrival seeds and at least N128.')
                new.append(c)
    write(path,dict(choices=choices,configs=[save_config(c) for c in new]));return new

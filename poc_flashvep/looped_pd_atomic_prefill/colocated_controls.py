"""Keep bounded colocated tuning available at newly qualified long-input cells."""
from .common import *
from .configs import finish
from .evidence import status,compare,full_work

CONTROL_KEYS=('token_budget','kv_blocks','wavefront','scheduler_mode','native_mixed')


def controls(cfg):
    return {key:cfg[key] for key in CONTROL_KEYS if key in cfg}


def qualification():
    from .search import configured_base
    new=[]
    for model,length,cap in [('huginn',3500,32),('ouro',16384,100)]:
        if model=='ouro':
            p=ROOT/'state/optional16k_gate.json'
            if not p.exists() or not json.loads(p.read_text()).get('passed'):continue
        # This is a bounded generic scheduler/memory control, not a new method.
        reference=configured_base(model,f'qual_{model}_colocated_control_reference_l{length}_o{cap}_v1',length,cap,
            phase='long_correctness',gpus=[0,1],D_token_budget=128,
            D_kv_blocks=28288 if model=='huginn' else 25088)
        new.append(finish(reference,8,rate=.5 if model=='huginn' else 1.))
        variants=[('separate_full','DP',length,None,None,False),
            ('separate_c1024','DP',1024,None,None,False),
            ('mixed_c256','DP_Mixed',256,None,None,True),
            ('mixed_compact_c256','DP_Mixed',256,2048 if model=='huginn' else 4096,28288 if model=='huginn' else 25088,False)]
        for index,(label,arm,chunk,budget,blocks,wave) in enumerate(variants):
            c=configured_base(model,f'qual_{model}_colocated_{label}_l{length}_o{cap}_v1',length,cap,
                arm=arm,chunk=chunk,chunk_label='full' if chunk==length else str(chunk),
                phase='long_correctness',gpus=[2*(index%3),2*(index%3)+1],wavefront=wave)
            if budget is not None:c.update(token_budget=budget,kv_blocks=blocks)
            if arm=='DP_Mixed':c.update(worker='native_mixed_worker',scheduler_mode='mixed',atomic_prompt=False,native_mixed=True)
            c=finish(c,8,rate=.5 if model=='huginn' else 1.)
            new.append(c)
        if model=='huginn':
            # The same supported wire-credit control already ran for Ouro.
            # Each new Huginn chunk still requires its own byte/output gate.
            for chunk in (2048,3500):
                c=configured_base(model,f'qual_huginn_native_credit1g_c{chunk}_l3500_o32_v1',3500,32,
                    phase='long_correctness',gpus=[4,5],chunk=chunk,chunk_label=str(chunk) if chunk<3500 else 'full',
                    D_token_budget=128,D_kv_blocks=28288,verify_transfer=True,
                    native_transport_limits=dict(max_inflight_bytes=1024**3))
                new.append(finish(c,8,rate=.5))
    for credit in (1024**3,8*1024**3):
        for chunk in (4096,8192):
            c=configured_base('ouro',f'qual_ouro_compact_D_credit{credit//1024**3}g_c{chunk}_l8192_o32_v1',8192,32,
                phase='long_correctness',gpus=[2,3],chunk=chunk,chunk_label='full' if chunk==8192 else str(chunk),
                D_token_budget=128,D_kv_blocks=25088,verify_transfer=True,
                native_transport_limits=dict(max_inflight_bytes=credit))
            new.append(finish(c,8,rate=1.))
    return new


def pilots():
    from .search import references,serving_config
    selected=references();gates={};new=[]
    for path in (ROOT/'configs').glob('qual_*_colocated_*_l*_o*_v1.json'):
        cfg=json.loads(path.read_text())
        if '_reference_' in cfg['name']:continue
        reference=f'qual_{cfg["model"]}_colocated_control_reference_l{cfg["length"]}_o{cfg["cap"]}_v1'
        result=compare(reference,cfg['name']) if status(reference)==status(cfg['name'])=='COMPLETE' else dict(passed=False,status=status(cfg['name']),reference_status=status(reference))
        if result['passed']:
            result['work']=full_work(cfg['name']);result['passed'] &= result['work']['passed']
        # An otherwise correct native Mixed run need not form a mixed batch
        # at every arrival rate. Record the actual composition separately.
        gates[cfg['name']]=result
        if not result['passed']:continue
        cell=selected[cfg['model']+'_L'+str(cfg['length'])]
        mu=min(max(v['mu_P'] for v in cell['chunks'].values()),cell['D'][str(cfg['cap'])]['mu_D'])
        label=cfg['name'].split('_colocated_',1)[1].split('_l',1)[0]
        for factor in (.75,1.,1.25):
            name=f'pilot_{cfg["model"]}_l{cfg["length"]}_o{cfg["cap"]}_colocated_{label}_f{str(factor).replace(".","p")}_v1'
            c=serving_config(cell,name,cfg['cap'],cfg['chunk'],'qps_pilot',24,rate=mu*factor,arm=cfg['arm'])
            c.update(controls(cfg),worker=cfg['worker'],rate_factor=factor,reference_capacity=mu,
                control_scope='Bounded existing colocated scheduler tuning at a newly qualified cell. Full BF16 depth and numerical fixes remain unchanged. Not a new mechanism.')
            new.append(c)
    for chunk in (2048,3500):
        name=f'qual_huginn_native_credit1g_c{chunk}_l3500_o32_v1'
        reference='qual_huginn_colocated_control_reference_l3500_o32_v1'
        result=compare(reference,name) if status(name)==status(reference)=='COMPLETE' else dict(passed=False,status=status(name))
        if result['passed']:
            from .evidence import transfer
            result.update(work=full_work(name),bytes=transfer(name))
            result['passed'] &= result['work']['passed'] and result['bytes']['passed']
        gates[name]=result
        if not result['passed']:continue
        cell=selected['huginn_L3500'];mu=min(max(v['mu_P'] for v in cell['chunks'].values()),cell['D']['32']['mu_D'])
        for factor in (.75,1.,1.25):
            name=f'pilot_huginn_l3500_o32_c{chunk}_credit1g_d128_kv28288_f{str(factor).replace(".","p")}_v1'
            c=serving_config(cell,name,32,chunk,'qps_pilot',24,rate=mu*factor)
            c.update(D_token_budget=128,D_kv_blocks=28288,native_transport_limits=dict(max_inflight_bytes=1024**3),
                rate_factor=factor,reference_capacity=mu,
                control_scope='Supported native NIXL wire-credit control. Separate from token chunking and Loop-specific mechanisms.')
            new.append(c)
    for credit in (1024**3,8*1024**3):
        for chunk in (4096,8192):
            name=f'qual_ouro_compact_D_credit{credit//1024**3}g_c{chunk}_l8192_o32_v1'
            reference='qual_ouro_d128_kv25088_l8192_o32_v1'
            result=compare(reference,name) if status(name)==status(reference)=='COMPLETE' else dict(passed=False,status=status(name))
            if result['passed']:
                from .evidence import transfer
                result.update(work=full_work(name),bytes=transfer(name))
                result['passed'] &= result['work']['passed'] and result['bytes']['passed']
            gates[name]=result
            if not result['passed']:continue
            cell=selected['ouro_L8192'];mu=max(v['mu_P'] for v in cell['chunks'].values())
            for factor in (.75,1.,1.25):
                name=f'pilot_ouro_l8192_o32_c{chunk}_credit{credit//1024**3}g_d128_kv25088_f{str(factor).replace(".","p")}_v1'
                c=serving_config(cell,name,32,chunk,'qps_pilot',24,rate=mu*factor)
                c.update(D_token_budget=128,D_kv_blocks=25088,native_transport_limits=dict(max_inflight_bytes=credit),
                    rate_factor=factor,reference_capacity=mu,
                    control_scope='Combined supported memory and transport controls. Separate from chunk choice and Loop-specific mechanisms. Each combination has its own byte/output gate.')
                new.append(c)
    write(ROOT/'state/colocated_extra_gates.json',gates)
    return new

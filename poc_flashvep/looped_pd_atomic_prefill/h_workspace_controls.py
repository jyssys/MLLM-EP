"""Gate the legal four-reservation Huginn memory control before Serving."""
from .common import *
from .configs import finish
from .evidence import status,compare,full_work,transfer

STAGE=ROOT/'state/H3500_workspace_control_stage.json'


def advance():
    from .search import configured_base,serving_config,references
    if not STAGE.exists():
        smokes={}
        for label in ('d128','d128_kv28288'):
            name=f'qual_huginn_{label}_l3500_memory_v1'
            if status(name)!='COMPLETE':return
            r=compare('qual_huginn_reference_l3500_memory_v1',name);r['bytes']=transfer(name)
            r['passed'] &= r['bytes']['passed'];smokes[label]=r
        if not all(v['passed'] for v in smokes.values()):
            write(STAGE,dict(stage='BLOCKED_MEMORY_SMOKE',smokes=smokes));return
        variants=[('reference',3500,None,None),('d128',3500,128,24576),('d128_kv28288',3500,128,28288),
            ('d128_kv28288_c256',256,128,28288),('d128_kv28288_c1024',1024,128,28288),('d128_kv28288_c2048',2048,128,28288)]
        new=[]
        for i,(label,chunk,budget,blocks) in enumerate(variants):
            c=configured_base('huginn',f'qual_huginn_{label}_l3500_o32_n8_v1',3500,32,chunk=chunk,
                chunk_label='full' if chunk==3500 else str(chunk),phase='long_correctness',gpus=[2*(i%3),2*(i%3)+1])
            if budget:c.update(D_token_budget=budget,D_kv_blocks=blocks)
            c=finish(c,8,rate=.6);c['P_slots']=2 if chunk==256 else 1;new.append(c)
        enqueue(new);write(STAGE,dict(stage='QUALIFICATION8',configs=[c['name'] for c in new],smokes=smokes));return
    state=json.loads(STAGE.read_text())
    if state['stage']=='QUALIFICATION8':
        if any(status(n) not in ('COMPLETE','FAILED') for n in state['configs']):return
        reference='qual_huginn_reference_l3500_o32_n8_v1';gates={}
        for name in state['configs'][1:]:
            r=compare(reference,name) if status(name)=='COMPLETE' else dict(passed=False,status=status(name))
            if r['passed']:r['work']=full_work(name);r['passed'] &= r['work']['passed']
            gates[name]=r
        new=[]
        for chunk in (256,1024,2048):
            gate=gates[f'qual_huginn_d128_kv28288_c{chunk}_l3500_o32_n8_v1']
            if not gate['passed']:continue
            c=configured_base('huginn',f'capacity_huginn_l3500_p_chunk{chunk}_control_v1',3500,32,chunk=chunk,chunk_label=str(chunk),
                worker='capacity_worker',phase='independent_capacity',capacity_mode='P_ONLY',gpus=[0])
            c=finish(c,8);c['jobs']=[dict(batch=b,cap=1,repeats=3) for b in (1,2,3,4)];new.append(c)
        enqueue(new);state.update(stage='CHUNK_CAPACITY',capacity_configs=[c['name'] for c in new],gates=gates);write(STAGE,state);return
    if state['stage']=='CHUNK_CAPACITY':
        if any(status(n) not in ('COMPLETE','FAILED') for n in state['capacity_configs']):return
        cell=references()['huginn_L3500'];mu=min(max(r['mu_P'] for r in cell['chunks'].values()),cell['D']['32']['mu_D']);new=[]
        for key,gate in state['gates'].items():
            if not gate['passed']:continue
            source=json.loads((ROOT/'configs'/(key+'.json')).read_text());chunk=source['chunk']
            label='d128' if source['D_kv_blocks']==24576 else 'd128_kv28288'
            suffix='full' if chunk==3500 else str(chunk)
            for factor in (.5,.75,1.,1.25):
                c=serving_config(cell,f'pilot_huginn_l3500_o32_c{suffix}_{label}_f{str(factor).replace(".","p")}_v1',32,chunk,'qps_pilot',24,rate=mu*factor)
                c.update(D_token_budget=128,D_kv_blocks=source['D_kv_blocks'],reference_capacity=mu,rate_factor=factor,
                    control_scope='Byte-qualified D workspace control. Four D reservations are possible only in the larger pool. Same graph geometry, BF16, full R32, and native handoff.')
                new.append(c)
        enqueue(new);state.update(stage='SERVING_PLANNED',configs=[c['name'] for c in new]);write(STAGE,state)


def enqueue(configurations):
    if not configurations:return
    names=[save_config(c) for c in configurations];p=ROOT/'configs/study_plan.json';old=json.loads(p.read_text())['configs']
    write(p,dict(configs=list(dict.fromkeys(names+old))))


if __name__=='__main__':advance()

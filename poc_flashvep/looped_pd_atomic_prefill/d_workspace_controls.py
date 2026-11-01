"""Expand only byte-qualified native Decode workspace and KV settings."""
from .common import *
from .evidence import compare,full_work,transfer,status
from .search import serving_config


def main():
    reference='qual_ouro_native_transport_default_l8192_o32_v1';gates={};new=[]
    cell=json.loads((ROOT/'configs/capacity_references.json').read_text())['ouro_L8192']
    mu=max(r['mu_P'] for r in cell['chunks'].values())
    for label,blocks in [('d128',22528),('d128_kv25088',25088)]:
        name=f'qual_ouro_{label}_l8192_o32_v1'
        if status(reference)!=status(name) or status(name)!='COMPLETE':gates[label]=dict(passed=False,status=status(name));continue
        result=compare(reference,name);result['work']=full_work(name);result['transfer']=transfer(name)
        result['passed'] &= result['work']['passed'] and result['transfer']['passed'];gates[label]=result
        if not result['passed']:continue
        chunks=(8192,4096) if blocks==22528 else (256,1024,2048,4096,8192)
        for chunk in chunks:
            for factor in (.75,1.,1.25):
                suffix='full' if chunk==8192 else str(chunk)
                c=serving_config(cell,f'pilot_ouro_l8192_o32_c{suffix}_{label}_f{str(factor).replace(".","p")}_v1',32,chunk,'qps_pilot',24,rate=mu*factor)
                c.update(D_token_budget=128,D_kv_blocks=blocks,rate_factor=factor,reference_capacity=mu,
                    control_scope='Existing native scheduler/cache configuration. P budget unchanged. D workspace right-sized before changing D KV capacity. Same Graph8, original weights, BF16 KV, R4, and NIXL protocol.')
                new.append(save_config(c))
    write(ROOT/'state/D_workspace_control_gates.json',gates)
    if new:
        p=ROOT/'configs/study_plan.json';old=json.loads(p.read_text())['configs'];write(p,dict(configs=list(dict.fromkeys(new+old))))
    print('D_WORKSPACE_CONTROL_TIMING',len(new))


if __name__=='__main__':main()

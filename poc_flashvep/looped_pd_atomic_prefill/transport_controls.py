"""Qualify supported wire-credit controls before their clean timing screen."""
from .common import *
from .evidence import compare,full_work,transfer,status
from .search import configured_base
from .configs import finish


def main():
    reference='qual_ouro_native_transport_default_l8192_o32_v1';gates={};new=[]
    cell=json.loads((ROOT/'configs/capacity_references.json').read_text())['ouro_L8192']
    strongest=max(r['mu_P'] for r in cell['chunks'].values())
    for label,limit in [('credit1g',1024**3),('credit8g',8*1024**3)]:
        name=f'qual_ouro_native_transport_{label}_l8192_o32_v1'
        if status(reference)!=status(name) or status(name)!='COMPLETE':
            gates[label]=dict(passed=False,status=status(name));continue
        result=compare(reference,name);result['work']=full_work(name);result['transfer']=transfer(name)
        result['passed'] &= result['work']['passed'] and result['transfer']['passed'];gates[label]=result
        if not result['passed']:continue
        for factor in (.5,1.,1.25,1.5):
            rate=strongest*factor
            for chunk in (8192,4096):
                suffix='full' if chunk==8192 else str(chunk)
                c=configured_base('ouro',f'pilot_ouro_l8192_o32_c{suffix}_{label}_f{str(factor).replace(".","p")}_v1',8192,32,
                    chunk=chunk,chunk_label=suffix,phase='qps_pilot',native_transport_limits=dict(max_inflight_bytes=limit))
                c=finish(c,24,rate=rate);c['P_slots']=4
                c.update(reference_capacity=strongest,rate_factor=factor,
                    capacity_reference=dict(mu_P=cell['chunks'][str(chunk)]['mu_P'],strongest_mu_P=strongest,mu_D=cell['D']['32']['mu_D']),
                    control_scope='Native PDConfig wire-credit tuning. Same transport protocol and 64-MiB fragments. Not a Loop Split improvement.')
                new.append(save_config(c))
    write(ROOT/'state/native_transport_credit_gates.json',gates)
    if new:
        path=ROOT/'configs/study_plan.json';old=json.loads(path.read_text())['configs']
        write(path,dict(configs=list(dict.fromkeys(new+old))))
    print('NATIVE_TRANSPORT_CONTROL_TIMING',len(new));return gates


if __name__=='__main__':main()

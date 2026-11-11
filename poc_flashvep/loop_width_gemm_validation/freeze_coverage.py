"""Extend the fixed map. Never reselect an existing geometry."""
from .common import *
from .micro_analysis import choices


def main():
    initialize();old=ROOT/'configs/operator_kernel_policy_s1.json'
    value=json.loads(old.read_text())
    value['parent_policy_SHA256']=digest(old)
    for model in ('huginn','ouro'):
        path=ROOT/'raw'/f'gemm_{model}_batch_coverage_s1/projection_results.json'
        data=json.loads(path.read_text());existing={(r['role'],r['M'],r['N'],r['K']) for r in value['models'][model]['entries']}
        assert data['metadata']['cublasLt_version']==value['models'][model]['cublasLt_version']
        for row in data['results']:
            key=(row['role'],row['rows'],*row['weight_shape']);assert key not in existing
            best,_=choices(row)
            if best['ms']>row['reference_graph']['p50_ms']*.98:best=dict(method='qualified_reference',ms=row['reference_graph']['p50_ms'])
            value['models'][model]['entries'].append(dict(role=key[0],M=key[1],N=key[2],K=key[3],**best))
        value['models'][model]['coverage_source']=dict(path=str(path),SHA256=digest(path))
    value['scope']='Fixed first-start map with additional batch coverage. Existing geometries keep every original algorithm choice. Independent starts 2 and 3 are confirmation, not selection.'
    target=ROOT/'configs/operator_kernel_policy_v2.json';assert not target.exists()
    write(target,value)
    qualification=[];policy_SHA=digest(target)
    for count in (8,32):
        for model in ('huginn','ouro'):
            arms=('zero','loop','token','token_group32','loop28','token1792') if model=='huginn' else ('zero','loop','token')
            for arm in arms:
                cfg=json.loads((ROOT/'configs'/f'kernel_{model}_{arm}_state{count}.json').read_text())
                cfg.update(name=f'kernel_v2_{model}_{arm}_state{count}',projection_kernel_policy='tuned_v2',
                    projection_kernel_policy_file=str(target),projection_kernel_policy_SHA256=policy_SHA,
                    workload=f'L{cfg["length"]}_O{cfg["cap"]}',
                    data_scope=dict(cfg.get('data_scope',{}),distinct_prompt_ids=count,cyclic_reuse=False),
                    D_prefill_cap_includes_prelude=arm.startswith('token'),
                    require_realized_quantum_contract=arm!='zero',
                    required_qualification=(f'kernel_v2_{model}_{arm}_state8' if count==32 else
                        None if arm=='zero' else f'kernel_v2_{model}_zero_state8'),
                    phase='BATCH_COVERED_KERNEL_MAP_EXACT_STATE_QUALIFICATION_NOT_TIMING')
                save(cfg);qualification.append(cfg['name'])
    write(ROOT/'state/v2_qualification_plan.json',dict(jobs=qualification,policy_SHA256=policy_SHA))
    print('QUALIFICATIONS',len(qualification),'POLICY',policy_SHA)


if __name__=='__main__':main()

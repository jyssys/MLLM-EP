"""Keep original-model token correctness separate from row-aware adapters."""
import json
import torch
from .common import ROOT,write
from .qualification import audit
from .qualification_gate import compare
from .quality_gate import compare as quality_compare


def done(name):
    p=ROOT/'raw'/name/'dispatcher_result.json'
    return p.exists() and json.loads(p.read_text())['complete']


def main():
    torch.set_num_threads(1);actions=[];results=[];mode0=[];full400=[]
    for cut in (1,2,4,8,12):
        teacher=f'Q13_StockFull36_Teacher{cut}_N8_v1';actual=f'Q13_StockToken{cut}_N8_FullState_v1';mixed=f'Q13_StockToken{cut}_C3_Natural32_v1'
        if cut==8:teacher,actual,mixed='Q8_StockFull36_TokenTeacher8_N8_v2','Q8_StockToken8_N8_FullState_v2','Q8_StockToken8_C3_Natural32_v2'
        if all(done(n) for n in (teacher,actual,mixed)):
            strict=compare(teacher,actual);small=audit(actual);c3=audit(mixed)
            qp=ROOT/'analysis'/(mixed+'_quality_gate.json');quality=json.loads(qp.read_text()) if qp.exists() else quality_compare(mixed)
            passed=strict['passed'] and small['passed'] and c3['passed'] and c3['decode_full36_observation_complete'] and c3['mixed_batches']>0 and quality['passed']
            results.append(dict(cut=cut,passed=passed,strict_N8=strict,mixed_N32=mixed,
                mixed_batches=c3['mixed_batches'],maximum_actual_D_batch=c3['maximum_actual_D_batch'],
                quality={k:v for k,v in quality.items() if k!='rows'}))
            if passed:actions.append(dict(axis='Token',cut=cut,suffix_fraction=None))
        m0=f'Q15_StockMode0_Token{cut}_N8_v1'
        if done(m0):
            strict0=compare(f'Q4_TokenTeacher{cut}_N8',m0);check=audit(m0)
            if strict0['passed'] and check['passed'] and check['decode_full36_observation_complete']:mode0.append(cut)
            write(ROOT/'analysis'/(m0+'_strict_comparison.json'),strict0)
    full='Q14_StockToken8_C3_Natural400_GPUdigest_v1'
    if done(full):
        proof=audit(full);qp=ROOT/'analysis'/(full+'_quality_gate.json')
        quality=json.loads(qp.read_text()) if qp.exists() else quality_compare(full)
        if proof['passed'] and proof['decode_full36_observation_complete'] and quality['passed'] and all(r['status']=='PASS' for r in proof['imported_prefix_KV']):full400.append(dict(axis='Token',cut=8,suffix_fraction=None))
    gate=dict(passed=bool(actions),qualified_axes=['Token'] if actions else [],qualified_actions=actions,
        stock_token_model=True,mixed_rows_passed=bool(actions),full400_runtime_stress_passed=bool(full400),
        full400_actions=full400,mode0_controlled_cuts=mode0,results=results,evidence='CORRECTNESS_ONLY',
        scope='Unmodified original model and Transformer layers. Default compiler plus prefix ownership and remote-count adapters only. '
            'Strict N8 uses a same-geometry original-model teacher. Natural32 uses the frozen task-quality gate. '
            'Only cut8 has the separate full400 stress when listed. Other cuts retain their N32 scope. '
            'Mode0 controlled compile interventions need the separate mode0 N8 proof. No performance or decode-safety claim.')
    write(ROOT/'state/C3_stock_token_gate_v1.json',gate)
    print('Stock-token actions:',actions,'Mode0 controls:',mode0,'Full400:',bool(full400),flush=True)
    return gate


if __name__=='__main__':main()

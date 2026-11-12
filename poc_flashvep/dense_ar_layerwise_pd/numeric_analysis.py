"""Evaluate preserved state tensors against the preregistered tolerance."""
import json,math
from .common import *
from .analyze import compare,records


def difference(reference,candidate):
    import torch
    assert reference.shape==candidate.shape and reference.dtype==candidate.dtype==torch.bfloat16
    a=reference.float();b=candidate.float();delta=(a-b).abs();limit=.02+.01*a.abs()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    bad=int((delta>limit).sum());total=a.numel()
    return dict(shape=list(a.shape),elements=total,finite=finite,bitwise_equal=bool(torch.equal(reference,candidate)),
        maximum_absolute_error=float(delta.max()),RMSE=float((delta.square().mean()).sqrt()),
        tolerance_exceedances=bad,tolerance_exceedance_fraction=bad/total,
        passed=finite and bad==0,atol=.02,rtol=.01)


def main():
    import torch
    torch.set_num_threads(1)
    names=json.loads((ROOT/'state/layer_numeric_plan.json').read_text())['jobs'];reference=names[0]
    results=[]
    for name,role,layers in ((names[1],'prefill',36),(names[2],'prefill',32),(names[2],'decode',36)):
        outputs=compare(reference,name);rows=[]
        for i in range(8):
            a_path=ROOT/'raw'/reference/'state_tensors'/f'prefill_q{i}.pt'
            b_path=ROOT/'raw'/name/'state_tensors'/f'{role}_q{i}.pt'
            a=torch.load(a_path,map_location='cpu',weights_only=True,mmap=True)
            b=torch.load(b_path,map_location='cpu',weights_only=True,mmap=True)
            assert len(a['KV'])==36 and len(b['KV'])==layers
            kv=[dict(layer=j,**difference(a['KV'][j],b['KV'][j])) for j in range(layers)]
            hidden=difference(a['hidden'],b['hidden']) if layers==36 else None
            rows.append(dict(request=f'q{i}',valid_KV=kv,final_hidden=hidden,
                reference_file_SHA256=digest(a_path),candidate_file_SHA256=digest(b_path),
                passed=all(x['passed'] for x in kv) and (hidden is None or hidden['passed'])))
        results.append(dict(name=name,role=role,layers=layers,outputs=outputs,requests=rows,
            passed=outputs['passed'] and all(x['passed'] for x in rows)))
    passed=all(r['passed'] for r in results)
    write(ROOT/'analysis/layer_N8_numerical_qualification.json',dict(passed=passed,results=results,
        contract=dict(atol=.02,rtol=.01,output_tokens='EXACT'),
        scope='Every valid queried KV cell and final Hidden of eight tested requests. Numeric equality is not bitwise equality. Migration equivalence to local split is a separate hash check.'))
    print(json.dumps(dict(passed=passed,results=[dict(name=r['name'],role=r['role'],passed=r['passed'],
        exact_outputs=r['outputs']['passed'],
        maximum_KV_error=max(k['maximum_absolute_error'] for q in r['requests'] for k in q['valid_KV']),
        KV_exceedances=sum(k['tolerance_exceedances'] for q in r['requests'] for k in q['valid_KV']),
        maximum_hidden_error=max((q['final_hidden']['maximum_absolute_error'] for q in r['requests'] if q['final_hidden']),default=None),
        hidden_exceedances=sum(q['final_hidden']['tolerance_exceedances'] for q in r['requests'] if q['final_hidden'])) for r in results]),indent=2))


if __name__=='__main__':main()

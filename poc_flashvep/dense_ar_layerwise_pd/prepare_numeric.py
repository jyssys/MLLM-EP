"""Retain tensors to test the preregistered diagnostic tolerance."""
import json
from .common import *
from .prepare import save


def main():
    initialize();old=json.loads((ROOT/'state/layer_recovery_plan.json').read_text());jobs=[]
    for name in old['jobs']:
        c=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        c.update(name=name.removesuffix('_v2')+'_numeric_v3',save_state_tensors=True,
            phase='LAYER_NUMERICAL_DIAGNOSTIC',
            numeric_contract=dict(atol=.02,rtol=.01,output_tokens='EXACT',
                reason='The local split matched all generated sequences but differed in KV hashes from Layer3 onward. Measure the actual error. Do not silently pass hash mismatches.'))
        jobs.append(save(c))
    write(ROOT/'state/layer_numeric_plan.json',dict(jobs=jobs,
        scope='Preserve full valid KV and final Hidden tensors for independent numeric comparison. This is instrumented correctness, not clean serving.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()

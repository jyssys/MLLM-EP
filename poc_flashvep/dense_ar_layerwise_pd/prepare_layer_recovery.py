"""Preserve failed warm-up geometry and test a legal short-prompt budget."""
import json
from .common import *
from .prepare import save


def main():
    initialize();old=json.loads((ROOT/'state/layer_initial_plan.json').read_text());jobs=[]
    for name in old['jobs']:
        c=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        c.update(name=name.removesuffix('_v1')+'_budget4096_v2',token_budget=4096,
            recovery_hypothesis='The failed Native warm-up used 32000 rows although P_slots*max_model_len was 4160. Test a legal 4096-row budget before attributing failure to Layer continuation.')
        jobs.append(save(c))
    write(ROOT/'state/layer_recovery_plan.json',dict(jobs=jobs,
        failure_scope='Original STATE_REFERENCE failed during compiled warm-up, before requests or Layer continuation. Root cause remains a hypothesis until the corrected matched control passes.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()

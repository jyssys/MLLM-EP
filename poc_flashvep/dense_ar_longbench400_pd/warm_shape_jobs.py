"""Falsify cold-shape explanations without changing the Primary protocol."""
import json
from .common import ROOT,write
from .run import save
from .next_jobs import plan


def main():
    jobs=[]
    suffixes=['Native']+[axis+str(cut) for cut in (2,4,8,12) for axis in ('Layer','Token')]
    for method in suffixes:
        original=ROOT/'configs'/f'MapC0_I0_O8_U0_{method}_v1.json'
        cfg=json.loads(original.read_text())
        cfg.update(name=f'WarmAll_I0_O8_U0_{method}_v1',gpus=[2,3],warmup_all_prompts=True,
            diagnostic_intervention='ADDITIONAL_WARMUP_OF_ALL_QUERY_GEOMETRIES',
            scope='Same prompts,arrival trace,output8,credit4 and actual1/1/1 as the original cell. '
                'The only planned intervention is an additional full prompt-pool warmup. '
                'This does not change the Primary dataset,settings or SLO. It is not an online method.')
        jobs.append(save(cfg))
    plan('warm_shape_input0_output8_v1',jobs)
    print('Frozen9 matched warm-shape counterfactuals.',flush=True)


if __name__=='__main__':main()

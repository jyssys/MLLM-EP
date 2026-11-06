"""Do not give same-depth batching an avoidable full-depth alignment wait."""
import time
from .prepare import *
from .run import execute


def main():
    initialize()
    write(ROOT/'state/same_depth_alignment_control.json',dict(time=time.time(),
        hypothesis='A waiting same-depth join can lose R-1 decode steps. One legal prefill step can align depth 0 with decode depth 1 instead.',
        rule='At most one alignment step per cohort, only when the current depth distance is one. No future information.',
        safety='This control can delay ready decode. Report its measured ITL and TPOT. Do not call it strict priority or a hard deadline guarantee.',
        fairness='All four controls share weights, placement, eight graph entries, warm-up, and a 512-token budget.'))
    for model in ('ouro','huginn'):
        for chunk in (128,248):
            reference=ROOT/'raw'/(f'micro_{model}_matched_c128_v4' if chunk==128 else f'micro_{model}_matched_c248_v1')/'micro_summary.json'
            while not reference.exists():
                if time.time()>=LONG_START_CUTOFF:return
                time.sleep(10)
            if not json.loads(reference.read_text())['all_micro_states_exact']:continue
            cfg=configuration(model)
            cfg.update(name=f'opportunity_{model}_c{chunk}_alignment_v1',worker='operator_micro',micro_opportunity=True,
                opportunity_chunk=chunk,opportunity_token_budget=512,opportunity_graph_rows=512,
                opportunity_strict_join=True,opportunity_same_align=True)
            save(cfg);execute(cfg['name'])


if __name__=='__main__':main()

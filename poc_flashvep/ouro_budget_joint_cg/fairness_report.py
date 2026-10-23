"""Render completed fairness evidence without inferring candidate quality."""
import csv
import json
from collections import defaultdict
import numpy as np
from .common import ROOT

def read(name):
    return list(csv.DictReader((ROOT/name).open()))

def main():
    parity=read('fairness_output_parity.csv');effects=read('fairness_paired_effects.csv')
    metrics=[r for r in read('serving_cohort_metrics.csv') if r['run_id'].startswith('cb_p1_fairness')]
    assert len(metrics)==160 and len(parity)==480
    text=['# Completed-output-token fairness','',
        'This ablation changes fairness only. It does not run a C/G optimizer.',
        'All five native paths cross stock and completion-aware fairness on each device.',
        'Atomic, staged G4, G2, and Split use C1024. Native Mixed uses C512.',
        'The queue contains eight model-process blocks and 160 measured observer cohorts.',
        'Each prompt-shape contrast has four independent GPU/process blocks and two input cohorts per block.',
        'The two source cohorts repeat across GPUs. This limits workload generalization.','',
        'All prompt work passed the Full-R4 accounting assertion.',
        'All 480 paired requests had the same generated token count and executed decode-loop work.',
        f'Output token IDs matched in {sum(r["tokens_exact"]=="True" for r in parity)}/480 pairs.',
        'The output differences do not establish an accuracy regression or prove equivalence.',
        'The new policy still requires full-answer quality evaluation.','',
        '| Path | Prompt/N | Stock mean cohort pooled ITL P99, ms | Completion-fair mean cohort pooled ITL P99, ms | Paired change, ms [95% CI] |',
        '| --- | --- | ---: | ---: | ---: |']
    for P,N in ((2033,4),(8177,8)):
        for policy in ('atomic','staged4','grouped','split','mixed'):
            values=[r for r in metrics if r['policy']==policy and int(r['P'])==P]
            stock=np.mean([float(r['ITL_pooled_P99_ms']) for r in values if r['fairness']=='stock'])
            fair=np.mean([float(r['ITL_pooled_P99_ms']) for r in values if r['fairness']=='completion'])
            effect=next(r for r in effects if r['policy']==policy and int(r['P'])==P and r['metric']=='ITL_pooled_P99_ms')
            text.append(f'| {policy} | {P}/{N} | {stock:.3f} | {fair:.3f} | {float(effect["mean_difference"]):.3f} [{float(effect["CI95_low"]):.3f}, {float(effect["CI95_high"]):.3f}] |')
    text+=['','The interval resamples process/GPU block aggregates, not individual tokens.',
        'The table reports means of within-cohort pooled quantiles. It is not a global pooled-token quantile.',
        'All these timings use the observer/shared-host lane, not a quiet clean-serving headline.',
        'A fairness effect must not be attributed to Joint C/G selection.',
        'P95, TTFT, and output-rate effects remain in `fairness_paired_effects.csv`.',
        'A lower P99 does not imply a lower P95, faster TTFT, or sustainable capacity.']
    (ROOT/'fairness.md').write_text('\n'.join(text)+'\n')
    pairs=[r for folder in (ROOT/'raw').glob('cb_p1_observer_control_*')
        for r in json.loads((folder/'observer_pairs.json').read_text())]
    total=sum(r['requests'] for r in pairs);exact=sum(r['output_exact'] for r in pairs)
    ratios=[r['event_wall_s']/r['clean_wall_s'] for r in pairs]
    (ROOT/'observer_controls.md').write_text('\n'.join([
        '# Ready-stream observer qualification','',
        'Two quiet GPU/model-process blocks compare clean and event lanes on all six native paths.',
        'Every lane uses the same completion fairness, C512, requests, and generation cap.',
        'Lane order reverses across source cohorts.',
        f'The output token IDs matched in {exact}/{total} request pairs.',
        'The single mismatch occurred in native Mixed at P2033/N4, trial 5.',
        'Its cause remains unresolved. The observer can change batch geometry through host timing.',
        'This possibility does not prove a numerical cause for this specific mismatch.',
        f'Individual cohort event/clean wall ratios ranged from {min(ratios):.6f} to {max(ratios):.6f}.',
        'The range is not a precise causal overhead estimate. Only two process blocks were measured.',
        'Event service data remain diagnostic. Clean client timing is a separate lane.',
        'No benchmark accuracy or capacity claim follows from this control.'])+'\n')
    print('Rendered fairness and observer qualification reports')

if __name__=='__main__':
    main()

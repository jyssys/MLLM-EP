"""Lossless producer-readiness diagnostic after the primary uncached campaign.

Not a new serving method, and never pooled with primary timing. Exact CPU
processor bytes/grid key an unchanged GPU vision+DeepStack output. Cached Full
is numerically compared to the uncached correctness reference before timing.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from poc_native_lp_ep4.contract import atomic_json, read_json
from poc_native_lp_ep4.scripts import finish_campaign as f
from poc_native_lp_ep4.scripts.check_correctness import errors
from poc_native_lp_ep4.scripts.analyze import clean_rows, summarize, trace_rows
from poc_native_lp_ep4.scripts.aggregate import component_waves, component_summary
from poc_native_lp_ep4.scripts.lifecycle import lifecycle_rows, summarize_lifecycle, dispatch_shape


def compare_uncached(cached, uncached):
    source=read_json(uncached/'worker.dp0.json')['waves']
    target=read_json(cached/'worker.dp0.json')['waves']
    events=[json.loads(line) for p in cached.glob('native*.jsonl') for line in p.open()]
    results=[]
    for wave in target:
        if wave['repeat']<0 or wave['arm']!='FULL':continue
        ref=next(w for w in source if w['arm']=='FULL' and
                 w['repeat']==wave['repeat'] and w['modality']==wave['modality'])
        a,b=ref['wave'],wave['wave']
        h=errors(torch.load(uncached/f'hidden.w{a}.dp0.pt',weights_only=True),
                 torch.load(cached/f'hidden.w{b}.dp0.pt',weights_only=True))
        l=errors(torch.load(uncached/f'logits.w{a}.dp0.pt',weights_only=True),
                 torch.load(cached/f'logits.w{b}.dp0.pt',weights_only=True))
        def answer(w):return [w['emitted'].get(str(r).split('-')[0]) for r in w['injection_ids']]
        tokens=answer(wave)==answer(ref)
        hits=[e for e in events if e.get('diagnostic')=='vision_cache_hit'
              and e.get('wave')==b and e.get('ep_rank') in (0,1)]
        hit_ok=wave['modality']!='VISUAL_PREFILL' or {e['ep_rank'] for e in hits}=={0,1}
        results.append({'modality':wave['modality'],'repeat':wave['repeat'],
            'hidden':h,'logits':l,'continuation_equal':tokens,
            'exact_processor_cache_hit_both_source_tp':hit_ok,
            'pass':h['relative_l2']<=.005 and l['max_abs']<=.25 and tokens and hit_ok})
    assert len(results)==6 and all(r['pass'] for r in results), results
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='v6')
    a=p.parse_args();assert a.version=='v6';f.VERSION=a.version
    root=Path('poc_native_lp_ep4')
    plan=read_json(root/'artifacts'/f'campaign_blocks_{a.version}.json')
    assert plan['status']=='MEASUREMENTS_COMPLETE'
    output=root/'artifacts'/f'producer_control_{a.version}.json'
    if output.exists():raise FileExistsError(output)
    best=plan['frozen_selection']['best_arm']
    os.environ['NATIVE_LP_VISION_CACHE']='1'
    correct=f.launch(f'cached_producer_correctness_{a.version}',['FULL',best],3,
        trace=True,light=True,correctness=True,timeout=1100)
    native=f.check_correctness(correct,f'cached_producer_validation_{a.version}.json')
    comparisons=compare_uncached(correct,root/'artifacts/runs/release_correctness_v4')
    atomic_json(root/'artifacts/correctness'/f'cached_vs_uncached_{a.version}.json',
        {'status':'PASS','scope':'LOSSLESS_VISION_OUTPUT_CONTROL_NOT_PRIMARY',
         'source_run':str(correct),'reference_run':str(root/'artifacts/runs/release_correctness_v4'),
         'rows':comparisons})
    clean=f.launch(f'cached_producer_clean_{a.version}',['CP512',best],3,
        paired=True,decode=128,timeout=1100)
    trace=f.launch(f'cached_producer_components_{a.version}',['CP512',best],3,
        trace=True,light=True,paired=True,decode=128,timeout=1100)
    layers=trace_rows(trace);life=lifecycle_rows(trace)
    atomic_json(output,{'status':'COMPLETE','evidence':'DIAGNOSTIC_NOT_PRIMARY',
        'cached_native_correctness_status':native['status'],
        'cached_vs_uncached_correctness':comparisons,
        'runs':[str(correct),str(clean),str(trace)],
        'clean_rows':clean_rows(clean),'clean_summary':summarize(clean_rows(clean)),
        'component_summary':component_summary(component_waves(layers)),
        'lifecycle_summary':summarize_lifecycle(life),'dispatch_shape':dispatch_shape(layers),
        'limits':['Separate restarted secondary lane with 3 repeats, not the 10-repeat primary.',
                  'Diagnostic cache lookup includes exact CPU input hashing; not a fastpath claim.',
                  'Local DeepEP prepare durations still include peer readiness, not isolated wire service.']})
    print('LOSSLESS_PRODUCER_CONTROL_COMPLETE',flush=True)


if __name__=='__main__':main()

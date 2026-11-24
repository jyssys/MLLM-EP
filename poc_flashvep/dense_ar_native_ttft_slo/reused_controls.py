"""Audit existing one-factor controls before requesting more GPU repeats."""
import json
import time
from .common import ROOT,PRIOR,REPO,write,digest,preserve
from .runtime import bind
from .reproduction import reconstruct
from .failure_analysis import save_csv
from .request_trace import trace,STAGES
from ..dense_ar_output_aware_v2.statistics import ratio,paired_difference


def analyze():
    runner=bind();preserve();runs=[];comparisons=[];sources={}
    plans=(('coordinator_hygiene_c3_confirmation_warm_regime_v10.json','GC',
        'OriginalHybrid','HybridGCControl','defer_coordinator_cyclic_gc'),
        ('c3_credit_retirement_confirmation_r1.json','Credit','OptimizedLayer','OriginalLayer',
        'source_credit_release_on_retirement'),
        ('c3_credit_retirement_confirmation_r1.json','Credit','OptimizedHybrid','OriginalHybrid',
        'source_credit_release_on_retirement'))
    for filename,family,left,right,factor in plans:
        plan=PRIOR/'jobs'/filename;sources[str(plan.relative_to(REPO))]=digest(plan)
        groups={}
        for block in json.loads(plan.read_text())['blocks']:
            if left not in block['arms'] or right not in block['arms']:continue
            pair=[]
            for label in (left,right):
                name=block['arms'][label];folder=PRIOR/'raw'/name
                cfg,metric,requests=reconstruct(folder);_,_,paths=trace(folder)
                receipt=json.loads((folder/'dispatcher_result.json').read_text());assert receipt['complete']
                value=dict(family=family,mode=block['mode'],start=block['start'],label=label,name=name,
                    cfg=cfg,signature=runner.signature(cfg),metrics=metric,
                    mean_host_ms={key:sum(row[key] for row in paths)/len(paths) for key in STAGES})
                pair.append(value)
                for item in ('configuration.json','dispatcher_result.json','requests.jsonl','window.json'):
                    path=folder/item;sources[str(path.relative_to(REPO))]=digest(path)
            a,b=pair
            for key in ('seed','gpus','dataset_SHA256','arrival_offsets','ignore_eos','output_cap',
                    'replay_output_lengths','source_capsule_SHA256'):
                assert a['cfg'].get(key)==b['cfg'].get(key),(filename,key)
            changes={key:[b['signature'].get(key),a['signature'].get(key)] for key in
                set(a['signature'])|set(b['signature']) if a['signature'].get(key)!=b['signature'].get(key)}
            assert changes=={factor:[False,True]},changes
            groups.setdefault(block['mode'],[]).append(pair)
            runs.append(dict(family=family,mode=block['mode'],start=block['start'],candidate=left,control=right,
                names=[a['name'],b['name']],factor=factor,changed_signature_fields=changes,
                GPU_pair=a['cfg']['gpus'],seed=a['cfg']['seed'],dataset_SHA256=a['cfg']['dataset_SHA256'],
                worker_source_SHA256=a['cfg']['source_capsule_SHA256'],
                actual_output_tokens=[row['metrics']['output_tokens'] for row in pair],
                evidence='REUSED_MEASURED_ONLINE_CONTROL',
                scope='Existing qualified campaign control. The pair shares GPU,source,arrival and output contract. '
                    'No new GPU measurement or Main400 attribution follows from this audit.'))
        for mode,pairs in groups.items():
            a,b=zip(*pairs)
            assert len(pairs)==3
            fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','generated_output_tokens_per_second',
                'SLO_qualified_output_tokens_per_second')
            comparisons.append(dict(family=family,mode=mode,candidate=left,control=right,factor=factor,
                paired_starts=len(pairs),ratios={field:ratio([row['metrics'][field] for row in a],
                    [row['metrics'][field] for row in b]) for field in fields},
                differences={field:paired_difference([row['metrics'][field] for row in a],
                    [row['metrics'][field] for row in b]) for field in
                    ('TTFT_mean_ms','Joint_SLO_attainment','every_ITL_SLO_attainment')},
                mean_host_deltas={field:paired_difference([row['mean_host_ms'][field] for row in a],
                    [row['mean_host_ms'][field] for row in b]) for field in STAGES},
                evidence='REUSED_MEASURED_ONLINE_CONTROL',
                scope='Pointwise paired start-level intervals. Prior Heldout192 controls remain distinct from '
                    'historical Main400 and fresh Main400. A GC effect here does not identify the cause of every old client-tail event.'))
    result=dict(time=time.time(),passed=True,runs=runs,comparisons=comparisons,source_files_SHA256=sources,
        scope='Read-only reuse of source-audited one-factor credit and coordinator-GC controls. '
            'Natural and equal replay remain separate. Original/Optimized labels follow each parent campaign,not a universal method. '
            'A timing effect does not imply joint-SLO or Native-relative decode success.')
    write(ROOT/'analysis/reused_one_factor_controls.json',result)
    save_csv(ROOT/'figures/reused_one_factor_control_per_start.csv',runs)
    save_csv(ROOT/'figures/reused_one_factor_control_contrasts.csv',comparisons)
    preserve();print('REUSED_ONE_FACTOR_CONTROLS',len(runs),'pairs',len(comparisons),'contrasts',flush=True)
    return result


if __name__=='__main__':analyze()

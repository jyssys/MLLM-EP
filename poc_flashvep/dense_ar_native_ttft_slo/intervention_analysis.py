"""Compare qualified policy contrasts and keep decode gates independent."""
import json
import time
import numpy as np
from .common import ROOT,write,digest,preserve
from .runtime import bind
from .registry import current_runs
from .reproduction import reconstruct
from .failure_analysis import save_csv
from .request_trace import trace,STAGES
from ..dense_ar_output_aware_v2.statistics import ratio,paired_difference,safety


PAIRS=(('StockCap2','StockToken'),('StockCap2RegOpen','StockCap2'),
    ('NativeRegOpen','NativeCommon2'),('NativeCommon0','NativeCommon2'),
    ('LayerRegOpen','Layer'),('HybridRegOpen','Hybrid'),
    ('LayerRetirement','Layer'),('HybridRetirement','Hybrid'))


def common_reference(label):
    if label=='StockToken' or label=='NativeCommon0':return 'NativeCommon0'
    if label.endswith('RegOpen') or label=='NativeRegOpen':return 'NativeRegOpen'
    return 'NativeCommon2'


def analyze():
    runner=bind();preserve();runs=[];by_key={}
    for item in current_runs():
        name=item['name'];folder=ROOT/'raw'/name
        cfg,metric,requests=reconstruct(folder)
        _,_,host=trace(folder)
        score_path=ROOT/'analysis'/(name+'_quality_gate.json')
        score=json.loads(score_path.read_text()) if score_path.exists() else dict(passed=False,status='NOT_EVALUATED')
        metric['decode_SLO_attainment']=float(np.mean([row['ITL_PASS'] and row['TPOT_PASS'] for row in requests]))
        value=dict(family=item['family'],label=item['label'],start=item['start'],name=name,
            signature=runner.signature(cfg),metrics=metric,quality_passed=score['passed'],
            quality_score=score.get('CI95_percentage_points'),source_capsule_SHA256=cfg['source_capsule_SHA256'],
            source_credit=cfg.get('continuous_source_credit_limit',0),D_cap=cfg.get('D_running_admission_cap'),
            config_SHA256=digest(folder/'configuration.json'),seed=cfg['seed'],GPU_pair=cfg['gpus'],
            dataset_SHA256=cfg['dataset_SHA256'],mode=item['mode'],
            mean_host_path_ms={field:float(np.mean([row[field] for row in host]))
                for field in (*STAGES,'Arrival_to_P_ready_ms','post_P_to_first_client_ms')})
        runs.append(value);by_key[item['family'],item['label'],item['start']]=value
    contrasts=[];gates=[]
    for family in sorted({row['family'] for row in runs}):
        labels=sorted({row['label'] for row in runs if row['family']==family})
        pairs=list(PAIRS)
        pairs+= [(label,reference) for label in labels for reference in
            tuple(dict.fromkeys(('NativeStrong',common_reference(label),'StockToken')))
            if reference!=label]
        for left,right in dict.fromkeys(pairs):
            indices=sorted(index for (fam,label,index) in by_key if fam==family and label==left
                and (family,right,index) in by_key)
            if not indices:continue
            a=[by_key[family,left,index] for index in indices]
            b=[by_key[family,right,index] for index in indices]
            for x,y in zip(a,b):
                for key in ('seed','GPU_pair','dataset_SHA256','mode'):assert x[key]==y[key]
            diffs={key:[b[0]['signature'].get(key),a[0]['signature'].get(key)]
                for key in set(a[0]['signature'])|set(b[0]['signature'])
                if a[0]['signature'].get(key)!=b[0]['signature'].get(key)}
            one_factor=(left,right) in PAIRS
            if one_factor:assert len(diffs)==1,(left,right,diffs)
            quality=all(row['quality_passed'] for row in a+b)
            fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','generated_output_tokens_per_second',
                'SLO_qualified_output_tokens_per_second')
            contrasts.append(dict(family=family,candidate=left,control=right,paired_starts=len(indices),
                start_indices=indices,quality_passed=quality,one_signature_factor=one_factor,changed_signature_fields=diffs,
                ratios={key:ratio([row['metrics'][key] for row in a],[row['metrics'][key] for row in b]) for key in fields},
                differences={key:paired_difference([row['metrics'][key] for row in a],[row['metrics'][key] for row in b])
                    for key in ('TTFT_mean_ms','TTFT_SLO_attainment','every_ITL_SLO_attainment','Joint_SLO_attainment')},
                mean_host_path_differences={key:paired_difference([row['mean_host_path_ms'][key] for row in a],
                    [row['mean_host_path_ms'][key] for row in b]) for key in a[0]['mean_host_path_ms']},
                actual_output_tokens_candidate=[row['metrics']['output_tokens'] for row in a],
                actual_output_tokens_control=[row['metrics']['output_tokens'] for row in b],
                evidence='MEASURED_ONLINE' if quality else 'FAILED_OR_PENDING_QUALITY_GATE',
                scope='One-factor status applies to exact runtime signatures,not to realized batch equality. '
                    'Natural outputs can differ after a policy changes batching. Equal-output replay controls their lengths separately. '
                    'Pointwise paired start-level empirical intervals are unadjusted. Fewer than3 starts has no CI. '
                    'Host-stage effects include feedback through later queues. Do not add them to kernel measurements from another workload.'))
        for label in labels:
            if label.startswith('Native'):continue
            reference=common_reference(label)
            indices=sorted(index for fam,name,index in by_key if fam==family and name==label
                and (family,'NativeStrong',index) in by_key and (family,reference,index) in by_key)
            if not indices:continue
            arm=[by_key[family,label,index] for index in indices]
            refs={name:[by_key[family,key,index] for index in indices]
                for name,key in (('NativeStrong','NativeStrong'),('NativeCommon',reference))}
            gate=safety([row['metrics'] for row in arm],{name:[r['metrics'] for r in rows] for name,rows in refs.items()})
            quality=all(row['quality_passed'] for row in arm+[r for rows in refs.values() for r in rows])
            gates.append(dict(family=family,label=label,common_D_policy_reference=reference,paired_start_indices=indices,
                quality_passed=quality,decode_gate=gate,qualified=bool(quality and gate['qualified']),
                common_P_source_credit_matched=all(row['source_credit']==refs['NativeCommon'][i]['source_credit'] for i,row in enumerate(arm)),
                scope='The Common Native matches D cap,registration and remote-READ/compute policy. '
                    'Source windows remain explicit. An unmatched source window does not become a common-P-policy control. '
                    'Absolute request decode attainment and Native-relative tail noninferiority are separate gates.'))
    value=dict(time=time.time(),passed=True,starts=runs,contrasts=contrasts,decode_gates=gates,
        evidence='FRESH_SEPARATE_FAMILY_ONLINE_POLICY_ANALYSIS',
        scope='Keep historical Main400,Natural current Main400 and Heldout192 equal replay separate. '
            'Correctness and task quality do not imply decode safety. A failed gate does not invalidate its diagnostic timing. '
            'The study identifies tested policy total effects. It does not force all TTFT regression into independent additive causal percentages.')
    write(ROOT/'analysis/controlled_intervention_analysis.json',value)
    for name,rows in (('controlled_intervention_per_start',runs),('controlled_intervention_contrasts',contrasts),
            ('controlled_intervention_decode_gates',gates)):
        save_csv(ROOT/'figures'/(name+'.csv'),rows)
    preserve();return value


if __name__=='__main__':
    value=analyze();print('CONTROLLED_INTERVENTION_ANALYSIS',len(value['starts']),'starts',len(value['contrasts']),'contrasts',flush=True)

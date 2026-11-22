"""Keep adaptive decode repair separate from the original frozen main test."""
import json
import time
import numpy as np
from .common import ROOT,REPO,write
from .confirmation_analysis import ratio
from .metrics import summarize
from .service_accounting import compare as boundaries
from .decode_tail_audit import audit as tail
from .final_confirmation_analysis import complete_metric


FIELDS=('TTFT_p95_ms','ITL_p99_ms','max_ITL_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second',
    'generated_output_tokens_per_second','Joint_SLO_attainment')


def load(name,source):
    folder=ROOT/'raw'/name
    path=folder/'dispatcher_result.json'
    if not path.exists():return None,None,'NOT_RUN'
    result=json.loads(path.read_text())
    if not result['complete']:return None,None,'FAILED'
    assert result['source_SHA256']==source
    cfg=json.loads((folder/'configuration.json').read_text())
    path=ROOT/'analysis'/(name+'_metrics.json')
    metric=complete_metric(json.loads(path.read_text()),name) if path.exists() else summarize(name)
    quality=ROOT/'analysis'/(name+'_quality_gate.json')
    if not quality.exists():return None,cfg,'PENDING_TASK_QUALITY'
    if not json.loads(quality.read_text())['passed']:return metric,cfg,'FAILED_TASK_QUALITY'
    assert cfg['count']==400 and not cfg['ignore_eos'] and not cfg['controlled_pipeline']
    return metric,cfg,'MEASURED_ONLINE'


def main():
    plan=json.loads((ROOT/'configs/hybrid_READ_repair_frozen_v1.json').read_text())
    extension=ROOT/'configs/hybrid_READ_repair_layer_control_extension_v1.json'
    if extension.exists():
        extra=json.loads(extension.read_text())
        assert extra['source_capsule_SHA256']==plan['source_capsule_SHA256']
        for addition in extra['cells']:
            cell=next(c for c in plan['cells'] if c['stage']=='INDEPENDENT_REPAIR_CONFIRMATION' and c['seed']==addition['seed'])
            assert all(cell[k]==addition[k] for k in ('QPS','start','GPU_pair','dataset'))
            cell['jobs'].update(addition['jobs'])
    groups=[]
    for stage in ('OBSERVED_TRACE_DIAGNOSTIC','INDEPENDENT_REPAIR_CONFIRMATION'):
        cells=[c for c in plan['cells'] if c['stage']==stage]
        arms=list(cells[0]['jobs']);values={a:[] for a in arms};cfgs={a:[] for a in arms};starts=[]
        for cell in cells:
            statuses={}
            for arm,name in cell['jobs'].items():
                metric,cfg,status=load(name,plan['source_capsule_SHA256'])
                statuses[arm]=dict(name=name,status=status)
                if status=='MEASURED_ONLINE':values[arm].append(metric);cfgs[arm].append(cfg)
            starts.append(dict(seed=cell['seed'],GPU_pair=cell['GPU_pair'],arms=statuses))
        complete=all(len(v)==len(cells) for v in values.values())
        row=dict(stage=stage,required_independent_starts=len(cells),starts=starts,
            status='MEASURED_ONLINE' if complete else 'PARTIAL_OR_NOT_RUN',arms={},comparisons={})
        for arm,v in values.items():
            row['arms'][arm]=dict(valid_starts=len(v),means={f:float(np.mean([r[f] for r in v]))for f in FIELDS} if v else {},
                output_tokens=[r['output_tokens'] for r in v],actual_geometry=[r['actual_geometry'] for r in v],
                contracts=[dict(credit=c.get('continuous_source_credit_limit',0),
                    compute_deferral=c.get('d_compute_throttle',False),READ_deferral=c.get('defer_D_remote_setup',False),
                    completion_guard=c.get('D_prefill_completion_guard',False),family=c['execution_family'])for c in cfgs[arm]])
        if complete:
            tests=[('HybridREAD',r)for r in ('NativeStrong','NativeREAD','HybridCompute')]
            tests += [('HybridCompute',r)for r in ('NativeStrong','NativeCompute')]
            if 'StockToken' in arms:
                tests += [('HybridREAD','StockToken'),('StockToken','NativeStrong'),('StockToken','NativeCompletion')]
            if 'LayerControl' in arms:
                tests += [('LayerControl','NativeStrong'),('LayerControl','NativeREAD'),('HybridREAD','LayerControl')]
            for arm,ref in tests:
                for a,b in zip(cfgs[arm],cfgs[ref]):
                    assert all(a[k]==b[k] for k in ('source_capsule_SHA256','dataset_SHA256','arrival_offsets',
                        'seed','gpus','output_cap','ignore_eos','token_budget','max_num_seqs'))
                comparisons={}
                for field in FIELDS:
                    aa=[r[field] for r in values[arm]];bb=[r[field] for r in values[ref]]
                    comparisons[field]=ratio(aa,bb) if len(cells)>=3 else dict(estimate=aa[0]/bb[0] if bb[0]>0 else None,
                        CI95=None,starts=1,qualifier='Post-observation single-start pilot. No safety CI.')
                gate=None if len(cells)<3 else all(comparisons[f]['CI95'][1]<=1.05 for f in ('ITL_p99_ms','TPOT_p95_ms'))
                row['comparisons'][arm+'_over_'+ref]=dict(ratios=comparisons,decode_5percent_CI_pass=gate,
                    absolute_decode_pass=all(r['decode_SLO_attainment']>=.95 for r in values[arm]))
            effects=[]
            for cell in cells:
                left,right=cell['jobs']['HybridREAD'],cell['jobs']['HybridCompute']
                a=json.loads((ROOT/'configs'/(left+'.json')).read_text())
                b=json.loads((ROOT/'configs'/(right+'.json')).read_text())
                from .final_confirmation_jobs import signature
                sa,sb=signature(a),signature(b)
                assert {k for k in sa if sa[k]!=sb[k]}=={'defer_D_remote_setup'}
                native=cell['jobs']['NativeREAD']
                effects.append(dict(seed=cell['seed'],host_boundaries=boundaries(left,right),
                    READ_arm_tail=tail(left,native)['summary'],control_tail=tail(right,native)['summary']))
            row['READ_only_intervention']=effects
        groups.append(row)
    result=dict(time=time.time(),groups=groups,source_capsule_SHA256=plan['source_capsule_SHA256'],
        complete=all(r['status']=='MEASURED_ONLINE' for r in groups),
        evidence='MEASURED_ONLINE_ADAPTIVE_REPAIR',
        scope='The original140-start main test and its unsafe outcomes remain unchanged. '
            'This follow-up was chosen after one main-test symptom. All new seeds and seven controls were fixed before new timing. '
            'Only READ setup deferral changes within fused Hybrid. Associated transfer and stream timing also change. '
            'Natural BI0 output counts and actual batches can differ. All task-quality gates and two Native safety controls remain. '
            'The observed-trace pilot has no CI. Confirmation uses3 paired independent starts,27 pointwise empirical bootstrap resamples. '
            'The additional frozen Layer control uses the same seeds and pair but follows the seven-arm blocks,not their randomized order. '
            'No exclusive host/DMA/kernel causal attribution,no global-best action claim,and no active-decode dynamic-window claim.')
    write(ROOT/'analysis/hybrid_READ_repair_results_v1.json',result)
    print('HYBRID_READ_REPAIR_ANALYSIS',[(r['stage'],r['status'])for r in groups],flush=True)


if __name__=='__main__':
    main()

"""Analyze both predefined geometry controls and retain every outcome."""
import json
import time
from .common import ROOT,bind,write,digest
from .critical_analysis import credits,csv_save
from .confirmation_analysis import FIELDS,P_STAGES


def start(name,label):
    bind()
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from ..dense_ar_output_aware_v2.c3_analysis import actual_quality
    from ..dense_ar_output_aware_v2.work_contract import executed_prompt_work
    from ..dense_ar_longbench400_pd.service_accounting import analyze as service
    from ..dense_ar_longbench400_pd.qualification import audit
    folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
    if not receipt.exists():return dict(name=name,label=label,status='NOT_RUN_OR_RUNNING')
    if not json.loads(receipt.read_text())['complete']:return dict(name=name,label=label,status='FAILED')
    cfg=json.loads((folder/'configuration.json').read_text());metric=summarize(name);quality=actual_quality(name)
    for role in ('prefill','decode'):
        runtime=json.loads((folder/f'actual_runtime_{role}.json').read_text())
        assert runtime['scheduler_B']==8192 and runtime['scheduler_S']==1024
        assert runtime['model_layers']==36 and runtime['parameter_dtypes']==['torch.bfloat16']
        assert json.loads((folder/f'KV_lifetime_{role}.json').read_text())['passed']
    if cfg.get('axis_kind'):
        coverage=audit(name);assert coverage['passed'] and coverage['decode_full36_observation_complete']
    path=service(name);data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    return dict(name=name,label=label,status='COMPLETE' if quality['passed'] else 'INVALID_QUALITY',
        quality_passed=quality['passed'],source_SHA256=cfg['source_capsule_SHA256'],
        dataset_SHA256=cfg['dataset_SHA256'],GPU_pair=cfg['gpus'],seed=cfg['seed'],QPS=cfg['qps'],
        B=8192,S=1024,KV_GiB=42,KV_block=128,axis_kind=cfg.get('axis_kind','Native'),
        cut=cfg.get('axis_cut'),fraction=cfg.get('hybrid_suffix_fraction'),
        source_credit=cfg.get('continuous_source_credit_limit',0),D_cap=cfg.get('D_running_admission_cap'),
        actual_geometry=metric['actual_geometry'],
        actual_executed_work=executed_prompt_work(cfg,folder,data),
        P_critical_mean_ms=sum(path['means'][key] for key in P_STAGES),host_path_ms=path['means'],
        source_window=credits(folder),**{key:metric[key] for key in FIELDS})


def analyze(freeze=False):
    bind();plan_path=ROOT/'jobs/p_balance_discovery208_v1.json';plan=json.loads(plan_path.read_text())
    rows=[start(item['name'],item['label']) for item in plan['arms']]
    by_label={r['label']:r for r in rows};complete=all(r['status'] in ('COMPLETE','INVALID_QUALITY','FAILED') for r in rows)
    candidates=[]
    for label,original in (('Layer8','OriginalLayer'),('HybridIso8Approx','OriginalHybrid')):
        value=by_label[label];control=by_label[original];stock=by_label['StockToken'];selection=None
        if all(v['status']=='COMPLETE' for v in (value,control,stock)):
            p=value['P_critical_mean_ms']/control['P_critical_mean_ms'];ttft=value['TTFT_p95_ms']/stock['TTFT_p95_ms']
            selection=dict(label=label,original=original,P_critical_ratio=p,TTFT_over_Stock=ttft,
                absolute_decode_attainment=value['decode_SLO_attainment'],
                selected=value['decode_SLO_attainment']>=.95 and (p<=.95 or ttft<=1.05),
                evidence='DISCOVERY_POINT_SCREEN_NOT_A_WIN')
        candidates.append(selection or dict(label=label,original=original,selected=False,status='INCOMPLETE_OR_INVALID'))
    result=dict(time=time.time(),complete=complete,plan_SHA256=digest(plan_path),starts=rows,screen=candidates,
        scope='One discovery start on the frozen208 requests. The heldout192 requests do not select a candidate. '
            'A screen pass is not a statistical win or a decode non-inferiority pass. '
            'Keep the original same-axis credit and D cap. Strong Stock remains untampered. '
            'Geometry changes redistribute real P/D work. They are not same-work compiler speedups.')
    write(ROOT/'analysis/p_service_balance_discovery.json',result)
    csv_save(ROOT/'figures/p_service_balance_discovery.csv',rows)
    csv_save(ROOT/'figures/p_service_balance_screen.csv',candidates)
    if freeze:
        assert complete,'All seven predefined outcomes must be present before freezing selection'
        target=ROOT/'state/p_service_balance_selection_frozen.json';assert not target.exists()
        write(target,dict(time=time.time(),analysis_SHA256=digest(ROOT/'analysis/p_service_balance_discovery.json'),
            discovery_dataset_SHA256=plan['dataset_SHA256'],candidates=candidates,
            selected=[v['label'] for v in candidates if v['selected']],
            reason='Apply the predeclared screen to both candidates. Freeze before generating heldout confirmation jobs.'))
    print('BALANCE_DISCOVERY',complete,candidates,flush=True)
    return result


if __name__=='__main__':
    import sys
    analyze(len(sys.argv)>1 and sys.argv[1]=='freeze')

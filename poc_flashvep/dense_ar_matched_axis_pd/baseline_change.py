"""Re-run only the new B/S on historical Native traces for audit."""
import json
import time
from .common import ROOT,PREVIOUS,REPO,write,digest,save
from .configuration import base
from .run import execute
from .analysis import metrics
from ..dense_ar_layer_token_pd.statistics_report import ratio_interval


def main():
    while not (ROOT/'state/discovery_complete.json').exists():
        time.sleep(15)
    groups=[]
    older=REPO/'poc_flashvep/reports/duetserve_agg_dp2_tp2_tbt_crossover_20261116'
    sources=[(PREVIOUS,f'heldout_Native_N128_O16_q6.4_s{seed}_v1') for seed in (2026111740,2026111741,2026111742)]
    sources.extend((older,f'confirm_Native_N128_O200_q6_s{seed}_v1') for seed in (2026111622,2026111623,2026111624))
    for historical,name in sources:
        old=json.loads((historical/'configs'/(name+'.json')).read_text())
        source_dataset=historical/'data'/old['dataset']
        data_name=f'bridge_O{old["cap"]}_{old["dataset"]}'
        data_path=ROOT/'data'/data_name
        if not data_path.exists():
            write(data_path,json.loads(source_dataset.read_text()))
        assert digest(data_path)==digest(source_dataset)
        reference=f'raw/HISTORICAL_bridge_{name}/requests.jsonl'
        target=ROOT/reference
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            # Preserve the original serialized rows, including their inputs.
            # File copying is an evidence operation, not a source rewrite.
            import shutil
            shutil.copyfile(historical/'raw'/name/'requests.jsonl',target)
        cfg=base('unified_bridge_'+name,'Native',0,old['count'],old['cap'],old['arrival_rate'],
            old['arrival_seed'],data_name,phase='HISTORICAL_NATIVE_CONFIGURATION_BRIDGE',
            arrival_offsets=old['arrival_offsets'],request_caps=old['request_caps'],
            historical_output_reference=reference,matrix_stage='baseline_bridge',
            common_prospective_KV_reservation=True,execution_contract_version=3)
        assert execute(save(cfg)),cfg['name']
        value=metrics(cfg['name'])
        path=historical/'analysis'/(name+'_metrics.json')
        before=json.loads(path.read_text())
        keys=('mean_TTFT_ms','TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s')
        groups.append(dict(cap=old['cap'],seed=old['arrival_seed'],QPS=old['arrival_rate'],
            historical=name,current=cfg['name'],count=cfg['count'],
            old_configuration_SHA256=digest(historical/'configs'/(name+'.json')),
            identical_arrival_offsets=True,identical_prompt_pool=True,
            old_settings={k:old.get(k) for k in ('token_budget','D_token_budget','P_slots','D_slots','P_group_size','D_admission_batch')},
            new_settings={k:cfg.get(k) for k in ('token_budget','D_token_budget','P_slots','D_slots','P_group_size','D_admission_batch')},
            historical_metrics={k:before[k] for k in keys},new_metrics={k:value[k] for k in keys},
            evidence='MEASURED',historical_reference=True,
            scope='Same prompt and arrival trace. B/S, cohort admission and source grouping change together. Hardware/software epochs also differ. Do not interpret this contrast as the isolated causal effect of B/S. No old B/S is launched in this campaign.'))
        write(ROOT/'analysis/native_historical_configuration_bridge.json',dict(starts=groups))
    comparisons=[]
    for cap in (16,200):
        rows=[r for r in groups if r['cap']==cap]
        comparisons.append(dict(cap=cap,starts=len(rows),metrics={key:ratio_interval(
            [r['historical_metrics'][key] for r in rows],[r['new_metrics'][key] for r in rows])
            for key in rows[0]['new_metrics']},scope='New divided by historical. Paired by historical trace. Not a clean simultaneous B-only intervention.'))
    write(ROOT/'analysis/native_historical_configuration_bridge.json',dict(starts=groups,comparisons=comparisons))
    write(ROOT/'state/baseline_bridge_complete.json',dict(time=time.time(),starts=len(groups)))


if __name__=='__main__':
    main()

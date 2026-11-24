"""Distinguish best-of-family actions from an actual iso-work comparison."""
import json
import time
from .common import ROOT,REPO,write,digest,preserve
from .runtime import bind
from .registry import analysis_runs
from .failure_analysis import save_csv


def analyze():
    bind();preserve()
    from ..dense_ar_output_aware_v2.work_contract import executed_prompt_work,validate
    result=[];sources={}
    for item in analysis_runs():
        if item['family'] not in ('Historical_Main400_Natural','Current_Main400_Natural',
                'Current_Heldout192_EqualReplay') or item['label'] in ('Agg','Deflection'):continue
        folder=item['root']/'raw'/item['name'];path=folder/'configuration.json'
        cfg=json.loads(path.read_text());data_path=item['root']/'data'/cfg['dataset']
        rows=json.loads(data_path.read_text());validate(cfg,rows)
        actual=executed_prompt_work(cfg,folder,rows)
        assert actual['observed_total_valid_prompt_token_layer_rows']>=actual['ideal_work_conserving_full36_rows']
        nominal=actual['nominal_static_partition']
        result.append(dict(family=item['family'],label=item['label'],start=item['start'],name=item['name'],
            GPU_pair=cfg['gpus'],seed=cfg['seed'],dataset_SHA256=cfg['dataset_SHA256'],
            axis_kind=cfg.get('axis_kind'),axis_cut=cfg.get('axis_cut'),suffix_fraction=cfg.get('hybrid_suffix_fraction'),
            structural_D_rows=nominal['D_rows'] if nominal else None,
            structural_P_rows=nominal['P_rows'] if nominal else 36*sum(r['input_tokens'] for r in rows[:cfg['count']]),
            inherited_iso_work_frontier_flag=cfg.get('iso_work_frontier'),
            inherited_target_D_rows=cfg.get('target_D_rows'),
            inherited_target_equals_structural=cfg.get('target_D_rows')==nominal['D_rows'] if nominal else None,
            work=actual,evidence='ACTUAL_QUERY_LAYER_TRACE_AND_STRUCTURAL_PARTITION',
            scope='Native,Stock Last8-equivalent,Layer Last2 and Hybrid2+1/8 are best-family actions,not equal offloaded rows. '
                'Legacy target labels do not override the prescribed and observed request partition. '
                'Structural attention pairs are not measured attention kernel time. Executed rows retain native proxy recomputation and preemption.'))
        for name in ('configuration.json','actual_queries_prefill.json','actual_queries_decode.json',
                'axis_events_prefill.json','axis_events_decode.json'):
            source=folder/name
            if source.exists():sources[str(source.relative_to(REPO))]=digest(source)
    value=dict(time=time.time(),passed=True,starts=result,source_files_SHA256=sources,
        scope='Validate exact request-local Token alignment,actual axis partitions and executed prompt rows. '
            'No fake iso-work claim follows from inherited target metadata. No GPU FLOP or exclusive-time estimate is derived from row counts.')
    write(ROOT/'analysis/native_relative_actual_work_audit.json',value)
    save_csv(ROOT/'figures/native_relative_actual_work.csv',result)
    preserve();print('NATIVE_RELATIVE_ACTUAL_WORK_AUDIT',len(result),'starts',flush=True);return value


if __name__=='__main__':analyze()

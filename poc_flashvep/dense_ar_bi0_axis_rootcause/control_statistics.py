"""Reuse every planned clean control. Do not count requests as server starts."""
import json
import numpy as np
from .common import ROOT,write
from .causal_replay import extract
from .statistics import interval
from .queue_accounting import audit


def summarize(name):
    source=extract(name);folder=ROOT/'raw'/name
    metric=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
    cfg=json.loads((folder/'configuration.json').read_text())
    assert not cfg.get('kernel_activity') and not cfg.get('correlated_profile')
    rows=source['rows']
    return dict(name=name,arrival_offsets=source['arrival_offsets'],dataset_SHA256=source['dataset_SHA256'],
        output_tokens=metric['output_tokens'],
        values=dict(P_service_mean_ms=1000*float(np.mean([r['P_service'] for r in rows])),
            D_first_mean_ms=1000*float(np.mean([r['D_first_service'] for r in rows])),
            P_plus_D_first_mean_ms=1000*float(np.mean([r['P_service']+r['D_first_service'] for r in rows])),
            old_serial_boundary_P_plus_D_mean_ms=float(np.mean([r['old_serial_boundary_P_plus_D_ms'] for r in rows])),
            TTFT_p95_ms=metric['TTFT_p95_ms'],TTFT_mean_ms=metric['TTFT_mean_ms'],
            goodput_tokens_per_s=metric['SLO_qualified_output_tokens_per_second']))


def main():
    groups=[]
    for mode in ('SERIAL_ADMISSION','ONLINE'):
        arms={axis:[f'CausalLast4_{axis}_{mode}_q2.35_c4_s2026112082_N200_v1',
            f'CausalPProfile_{axis}_{mode}_clean_after_s2026112082_N200_v1',
            f'RuntimeAPIProfile_{axis}_{mode}_clean_after_s2026112082_N200_v1'] for axis in ('Layer','Token')}
        groups.append(('FIXED_OUTPUT_'+mode,arms))
    groups.append(('NATURAL_EOS_ONLINE',{axis:[f'NaturalLast4Repro_{axis}_clean_before_q2.35_c4_s2026112061_N200_v1',
        f'NaturalLast4Repro_{axis}_clean_after_q2.35_c4_s2026112061_N200_v1',
        f'RuntimeAPIProfile_{axis}_NATURAL_REPRODUCTION_clean_after_s2026112061_N200_v1'] for axis in ('Layer','Token')}))
    result=[]
    for label,arms in groups:
        if not all((ROOT/'analysis'/(name+'_metrics.json')).exists() for names in arms.values() for name in names):continue
        data={axis:[summarize(name) for name in names] for axis,names in arms.items()}
        trace=data['Layer'][0]['arrival_offsets'];dataset=data['Layer'][0]['dataset_SHA256']
        assert all(r['arrival_offsets']==trace and r['dataset_SHA256']==dataset for values in data.values() for r in values)
        ratios={key:interval([r['values'][key] for r in data['Layer']],[r['values'][key] for r in data['Token']])
            for key in data['Layer'][0]['values']}
        queue=None
        if label=='NATURAL_EOS_ONLINE':
            queue={axis:[audit(ROOT/'raw'/name)['summary'] for name in names] for axis,names in arms.items()}
        result.append(dict(group=label,cut=4,N=200,clean_starts_per_axis=3,independent_arrival_seeds=1,
            source_controls=data,Layer_over_Token=ratios,source_queue_accounting=queue,
            evidence='MEASURED_ONLINE',scope='Post-processing of all3 clean starts specified by the profiler protocols. '
            'No profile start is pooled. Pairs use the same control phase and one fixed arrival trace. '
            'This tests process-start repeatability, not new-arrival-seed generalization. It is separate from the primary3-seed Natural EOS all-cuts confirmation.'))
    write(ROOT/'analysis/Last4_clean_control_start_statistics.json',dict(results=result))
    print(json.dumps([dict(group=r['group'],P_plus_D=r['Layer_over_Token']['P_plus_D_first_mean_ms'],
        TTFT=r['Layer_over_Token']['TTFT_p95_ms']) for r in result],indent=2))


if __name__=='__main__':main()

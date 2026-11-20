"""Keep same-cell TTFT accounting separate from cross-cell hypotheses."""
import json
import numpy as np
from .common import ROOT,PREVIOUS,write,digest


def historical():
    sides={}
    for axis in ('Layer','Token'):
        name=f'ControlledDiscovery_{axis}4_N200_q2.35_s2026112061_BI0_v3'
        path=PREVIOUS/'analysis'/(name+'_axis_timeline.json')
        data=json.loads(path.read_text());rows=data['rows']
        fields=['P_queue_ms','P_prepare_to_ready_ms','ready_to_D_prepare_ms',
            'D_prepare_to_first_worker_output_ms','first_worker_to_client_ms']
        sides[axis]=dict(name=name,source_SHA256=digest(path),
            mean_components={key:float(np.mean([r[key] for r in rows])) for key in fields},
            TTFT_mean_ms=float(np.mean([r['TTFT_ms'] for r in rows])),
            output_tokens=sum(r['output_tokens'] for r in rows),
            prompt_ids=[r['prompt_id'] for r in sorted(rows,key=lambda r:int(r['request_id'][1:]))])
    assert sides['Layer']['prompt_ids']==sides['Token']['prompt_ids']
    differences={key:sides['Layer']['mean_components'][key]-sides['Token']['mean_components'][key]
        for key in fields}
    differences['TTFT_mean_ms']=sides['Layer']['TTFT_mean_ms']-sides['Token']['TTFT_mean_ms']
    assert abs(sum(differences[key] for key in fields)-differences['TTFT_mean_ms'])<1e-3
    source=PREVIOUS/'analysis/BI0_axis_serial_component_comparison_v1.json'
    old=json.loads(source.read_text())['results']
    subset=[dict(cut=r['cut'],split=r['split'],N=r['N'],
        Layer_minus_Token_host_ms=r['Layer_minus_Token_component_mean_ms']) for r in old if r['cut'] in (4,8)]
    return dict(historical_Last4=sides,Layer_minus_Token_mean_ms=differences,
        serial_subset_audit=subset,evidence='HISTORICAL_TRACE_ANALYSIS',
        scope='The reported2674ms online gap is a mean-TTFT difference, not a p95 difference. '
        'The old25.35ms serial benefit is cut8/calibration20. Its cut8/validation15 counterpart has the opposite sign. '
        'These subset observations are not added to the cut4/full200 queue difference. Natural output counts can differ by numerical geometry.')


def main():
    value=historical();write(ROOT/'analysis/causal_historical_TTFT_partition.json',value)
    print(json.dumps({key:item for key,item in value.items() if key!='historical_Last4'},indent=2))


if __name__=='__main__':main()

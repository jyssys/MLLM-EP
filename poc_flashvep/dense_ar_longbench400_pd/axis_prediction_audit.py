"""Check winner prediction and regret without converting replay into online state."""
import json
import time
import numpy as np
from .common import ROOT,write
from .c0_queue_model import part,simulate


def main():
    model=json.loads((ROOT/'analysis/c0_host_service_model_v1.json').read_text())['model']
    actual=json.loads((ROOT/'analysis/axis_matrix_heldout_confirmation_r2.json').read_text())
    rows=[]
    for cell in actual['cells']:
        if cell['completed_paired_starts']!=cell['required_independent_starts']:
            continue
        measured={a:[] for a in ('Layer','Token')}
        predicted={a:[] for a in measured}
        for start in cell['results']:
            for axis in measured:
                name=start[axis]
                cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
                data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
                prediction=simulate(cfg['arrival_offsets'],[r['input_tokens'] for r in data],
                    [part(cfg,i,r['input_tokens']) for i,r in enumerate(data)],cfg['output_cap'],model)
                m=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
                measured[axis].append(m['TTFT_p95_ms'])
                predicted[axis].append(prediction['TTFT_p95_ms'])
        m={a:float(np.mean(v)) for a,v in measured.items()}
        p={a:float(np.mean(v)) for a,v in predicted.items()}
        chosen=min(p,key=p.get)
        label={'LAYER_CONFIRMED':'Layer','TOKEN_CONFIRMED':'Token'}.get(cell['status'])
        rows.append(dict(input_bucket=cell['input_bucket'],controlled_output=cell['output'],cut=cell['cut'],
            independent_starts=cell['completed_paired_starts'],measured_status=cell['status'],
            predicted_axis=chosen,measured_means_ms=m,predicted_means_ms=p,
            prediction_correct=None if label is None else chosen==label,
            latency_regret_percent=100*(m[chosen]/min(m.values())-1),
            actual_safety_valid=cell['status']!='INVALID'))
    clear=[r for r in rows if r['prediction_correct'] is not None]
    valid=[r for r in rows if r['actual_safety_valid']]
    result=dict(time=time.time(),rows=rows,clear_winner_cells=len(clear),
        correct_clear_winner_cells=sum(r['prediction_correct'] for r in clear),
        clear_winner_accuracy=sum(r['prediction_correct'] for r in clear)/len(clear),
        valid_latency_regret_cells=len(valid),
        mean_latency_regret_percent=float(np.mean([r['latency_regret_percent'] for r in valid])),
        p95_latency_regret_percent=float(np.percentile([r['latency_regret_percent'] for r in valid],95)),
        maximum_latency_regret_percent=max(r['latency_regret_percent'] for r in valid),
        evidence='OFFLINE_REPLAY',
        scope='The existing serial-calibrated state-updating model is not refit to winner labels. '
            'The simulation knows the full controlled arrival trace and output contract. It is not an online observable scalar. '
            'Clear labels use independent-start TTFT CIs and measured decode gates. Ties remain unlabeled. '
            'Latency regret is against the smaller measured mean start-level TTFTp95,not sustainable goodput or a causal oracle. '
            'Invalid safety cells stay in the ledger and do not enter the valid regret summary. '
            'A good full-model prediction does not establish a one-/two-knob online controller or C3 generalization.')
    write(ROOT/'analysis/c0_axis_winner_prediction_audit_v1.json',result)
    print({k:result[k] for k in ('clear_winner_cells','correct_clear_winner_cells','clear_winner_accuracy',
        'mean_latency_regret_percent','p95_latency_regret_percent','maximum_latency_regret_percent')},flush=True)


if __name__=='__main__':
    main()

"""Fit small nonnegative cost curves and test disjoint real prompts."""
import itertools
import json
import numpy as np
from .common import ROOT,write


def features(lengths):
    x=np.asarray(lengths,dtype=float)/8192.
    return np.column_stack((np.ones_like(x),x,x*x))


def fit(lengths,values):
    matrix=features(lengths);target=np.asarray(values,dtype=float);choices=[]
    for size in (1,2,3):
        for selected in itertools.combinations(range(3),size):
            beta=np.linalg.lstsq(matrix[:,selected],target,rcond=None)[0]
            if min(beta)<-1e-10:continue
            coefficient=np.zeros(3);coefficient[list(selected)]=np.maximum(beta,0.)
            choices.append((float(np.mean((matrix@coefficient-target)**2)),coefficient))
    assert choices
    return min(choices,key=lambda item:item[0])[1].tolist()


def predict(coefficients,length):
    return float(features([length])[0]@np.asarray(coefficients))


def error_summary(actual,predicted):
    actual=np.asarray(actual);predicted=np.asarray(predicted);residual=actual-predicted
    return dict(n=len(actual),MAE_ms=float(np.mean(np.abs(residual))),
        absolute_error_p95_ms=float(np.percentile(np.abs(residual),95)),
        absolute_error_p99_ms=float(np.percentile(np.abs(residual),99)),
        underprediction_frequency=float(np.mean(residual>0)),
        positive_error_p95_ms=float(np.percentile(np.maximum(residual,0),95)),
        signed_bias_prediction_minus_actual_ms=float(np.mean(-residual)),
        relative_absolute_error_mean=float(np.mean(np.abs(residual)/np.maximum(actual,1e-6))))


def main():
    fields=('P_prepare_to_ready_ms','P_serial_submit_to_ready_ms','ready_to_first_worker_output_ms')
    results={}
    for axis,cut in [('Native',0)]+[(a,c) for c in (1,2,4,6,8,12) for a in ('Layer','Token')]:
        names={split:f'ContinuationCost_{axis}{cut}_{split}{count}_BI0_v1'
               for split,count in (('calibration',20),('validation',15))}
        rows={split:json.loads((ROOT/'analysis'/f'{name}_axis_timeline.json').read_text())['rows'] for split,name in names.items()}
        assert not ({r['prompt_id'] for r in rows['calibration']}&{r['prompt_id'] for r in rows['validation']})
        for items in rows.values():
            for row in items:row['P_serial_submit_to_ready_ms']=row['P_queue_ms']+row['P_prepare_to_ready_ms']
        curves={}
        for field in fields:
            coefficients=fit([r['input_tokens'] for r in rows['calibration']],[r[field] for r in rows['calibration']])
            training=[predict(coefficients,r['input_tokens']) for r in rows['calibration']]
            validation=[predict(coefficients,r['input_tokens']) for r in rows['validation']]
            margins=[r[field]-p for r,p in zip(rows['calibration'],training)]
            margin=max(0.,float(np.percentile(margins,95)))
            curves[field]=dict(coefficients_ms=coefficients,training=error_summary([r[field] for r in rows['calibration']],training),
                validation=error_summary([r[field] for r in rows['validation']],validation),
                training_positive_residual_p95_ms=margin,
                validation_upper_bound_coverage=float(np.mean([p+margin>=r[field] for r,p in zip(rows['validation'],validation)])),
                scope='Nonnegative intercept, linear and quadratic terms in L/8192. Training residual margin is not WCET. Twenty calibration requests cannot establish a safe p99 bound.')
        results[f'{axis}{cut}']=dict(axis=axis,cut=cut,curves=curves,names=names)
    write(ROOT/'analysis/BI0_serial_service_model_v2.json',dict(actions=results,evidence='MODELED_ESTIMATE',model_version=2,
        scope='Independent prompt split. Host service envelopes include model, transport, submission and sampling. '
        'Source replay uses serial arrival/submission-to-ready, including IPC/admission. No other request can queue in these serial cost runs. '
        'No kernel-exclusive cost or safe online admission guarantee. Validation does not use future arrivals.'))
    return results


if __name__=='__main__':main()

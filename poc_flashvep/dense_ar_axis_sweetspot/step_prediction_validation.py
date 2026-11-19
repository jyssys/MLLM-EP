"""Validate frozen single-step and cumulative predictions on fresh traces."""
import json
import statistics
import time
from .common import ROOT,write,digest
from .historical_audit import records
from .measurement_export import quantile
from .replay import Cost,simulate


def error_summary(actual,predicted):
    errors=[1000*(p-a) for a,p in zip(actual,predicted)]
    assert errors
    return dict(samples=len(errors),signed_mean_ms=statistics.mean(errors),
        MAE_ms=statistics.mean(abs(x) for x in errors),
        p95_absolute_ms=quantile([abs(x) for x in errors],.95),
        p99_absolute_ms=quantile([abs(x) for x in errors],.99),
        underprediction_frequency=sum(x<0 for x in errors)/len(errors),
        nominal_plus_5pct_coverage=sum(a<=1.05*p for a,p in zip(actual,predicted))/len(errors))


def main(version=None):
    if version is None:
        for v in (2,3,4):
            if (ROOT/f'configs/frozen_action_service_costs_v{v}.json').exists():main(v)
        return
    modelpath=ROOT/f'configs/frozen_action_service_costs_v{version}.json'
    if not modelpath.exists():return
    receipt=ROOT/f'state/service_model_v{version}_frozen.json'
    if not receipt.exists():return
    assert json.loads(receipt.read_text())['SHA256']==digest(modelpath)
    model=json.loads(modelpath.read_text())['model']
    lookup={(r['method'],r['layers'],r['output'],r['batch']):Cost(**r['cost_seconds'])
        for r in model['models']}
    training={r['name'] for r in model['training']}
    stats=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    output=[]
    for run in stats['starts']:
        if version>=3 and run['stage']!='batch4_knob_confirmation':continue
        key=run['method'],run['layers'],run['output'],run['actual_batch']
        if key not in lookup or run['name'] in training:continue
        name=run['name'];c=lookup[key];b=run['actual_batch']
        host=json.loads((ROOT/'analysis'/(name+'_host_paths.json')).read_text())
        requests={'q'+str(r['sample_id']):r for r in records(ROOT/'raw'/name/'requests.jsonl')}
        P=[r['spans_ms']['P_submit_to_source_notification']/1000 for r in host['request_paths']]
        D=[r['last_client_notice']-r['D_host_submit'] for r in host['cohorts']]
        preparation=[max(requests[r]['first_token'] for r in co['requests'])-co['D_host_submit'] for co in host['cohorts']]
        predicted=simulate(run['arrival_offsets'],[c]*run['count'],b,run['source_window'])
        actualTTFT=[requests['q'+str(i)]['ttft_ms']/1000 for i in range(run['count'])]
        output.append(dict(name=name,stage=run['stage'],method=run['method'],layers=run['layers'],
            output=run['output'],batch=b,QPS=run['QPS'],seed=run['seed'],
            P_step=error_summary(P,[c.P]*len(P)),
            D_preparation_cohort=error_summary(preparation,[b*c.preparation]*len(preparation)),
            D_whole_cohort=error_summary(D,[b*c.preparation+c.decode]*len(D)),
            client_TTFT=error_summary(actualTTFT,[r['TTFT'] for r in predicted]),
            evidence='OFFLINE_REPLAY',
            scope='No actual future finish enters frozen costs or rule inputs. Actual finishes are used only as post-run validation targets. Samples are within-run observations, not independent server starts. The5% coverage diagnostic is not a calibrated upper bound or WCET.'))
    filename='frozen_model_step_and_completion_errors'+(f'_v{version}' if version>=3 else '')+'.json'
    write(ROOT/'analysis'/filename,dict(time=time.time(),
        model_version=version,model_SHA256=digest(modelpath),results=output,
        scope='Nominal calibration, individual host-step error and accumulated client-TTFT error stay separate. No cost refit or safety-margin tuning uses these evaluation results.'))
    print('STEP_PREDICTION_VALIDATION',version,len(output),flush=True)


if __name__=='__main__':
    main()

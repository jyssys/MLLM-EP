"""Compare serial Decode enclosures with delivery ITL. Do not infer safe slack."""
from collections import defaultdict
from .common import *
from .analyze import requests,table,pct


def analyze(model):
    name=f'p_{model}_native_serial8_v1';folder=ROOT/'raw'/name
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    outputs={r['seed']:r for r in requests(folder)};steps=defaultdict(list)
    for file in folder.glob('device_execution_*.json'):
        value=json.loads(file.read_text())
        if value['role']!='decode':continue
        for row in value['rows']:
            if not row['measured']:continue
            assert len(row['requests'])==1,'Serial diagnostic contains batched work'
            req=row['requests'][0]
            if req['seed'] in outputs and req['generated_tokens']>0:
                steps[(req['seed'],req['generated_tokens'])].append(row)
    result=[]
    for (seed,generated),operations in sorted(steps.items()):
        event=outputs[seed]['token_events'][generated]
        assert not event['first'] and event['new_tokens']==1
        recurrent=[r for r in operations if r['operation']=='recurrent']
        assert sum(r['loops'] for r in recurrent)==cfg['depth']
        duration=sum(r['stream_elapsed_ms'] for r in operations)
        itl=1000*(event['time']-event['previous_time'])
        result.append(dict(model=model,seed=seed,generated_before_step=generated,
            native_delivery_ITL_ms=itl,model_operation_event_sum_ms=duration,
            operation_count=len(operations),recurrent_submissions=len(recurrent),
            delivery_minus_enclosure_sum_ms=itl-duration,
            scope='Serial calibration only. The sum can contain stream waits. Its residual is not physical idle, online slack, or an available borrowing budget.'))
    table(ROOT/f'{model}_serial_decode_service_diagnosis.csv',result)
    return dict(model=model,run=name,steps=len(result),
        ITL_p50_ms=pct([r['native_delivery_ITL_ms'] for r in result],50),
        event_sum_p50_ms=pct([r['model_operation_event_sum_ms'] for r in result],50),
        residual_p50_ms=pct([r['delivery_minus_enclosure_sum_ms'] for r in result],50),
        physical_idle='NOT_MEASURED',usable_future_slack='NOT_MEASURED',
        scope='A loop longer than a physical idle gap does not by itself exclude deadline-safe overlap with host submission. This diagnostic does not establish such overlap.')


def main():
    initialize();result=[analyze(model) for model in ('ouro','huginn')]
    write(ROOT/'serial_decode_service_diagnosis.json',result);print(json.dumps(result))


if __name__=='__main__':
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    initialize()
    with quiet_analysis('serial_Decode_service_diagnostic'):main()

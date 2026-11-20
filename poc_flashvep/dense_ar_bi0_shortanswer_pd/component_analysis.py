"""Compare matched real-request host components without summing GPU claims."""
import json
import numpy as np
from .common import ROOT,write
from .axis_timeline import analyze
from .serving_analysis import stats


def data(axis,cut,split,count):
    name=f'ContinuationCost_{axis}{cut}_{split}{count}_BI0_v1';folder=ROOT/'raw'/name
    receipt=folder/'dispatcher_result.json'
    if not receipt.exists() or not json.loads(receipt.read_text())['complete']:return None
    trace=analyze(name)
    queries=json.loads((folder/'actual_queries_decode.json').read_text())['rows']
    source=json.loads((folder/'actual_queries_prefill.json').read_text())['rows']
    hidden=folder/'hidden_lifetime_decode.json'
    hidden_rows=json.loads(hidden.read_text())['rows'] if hidden.exists() else []
    hidden_rows=[r for r in hidden_rows if r['event']=='D_HIDDEN_READ_DONE' and r['request_id'].startswith('q')]
    setup=json.loads((folder/'NIXL_host_setup_decode.json').read_text())['rows']
    telemetry=[r for r in setup if r['kind']=='NIXL_native_telemetry' and str(r.get('request_id','')).startswith('q')
        and not r.get('agent_name','').startswith('BI0-hidden-')]
    rows=[]
    for item in trace['rows']:
        rid=item['request_id'];length=item['input_tokens']
        rows.append(dict(item,
            P_prefill_query_calls=sum(q['request_id']==rid for q in source),
            D_prompt_query_calls=sum(q['request_id']==rid and q['start']<length for q in queries),
            isolated_first_path_ms=item['P_prepare_to_ready_ms']+item['ready_to_first_worker_output_ms']))
    return dict(name=name,rows=rows,hidden=hidden_rows,KV_telemetry=telemetry)


def main():
    results=[]
    fields=('P_prepare_to_ready_ms','ready_to_D_prepare_ms','D_prepare_to_first_worker_output_ms',
        'ready_to_first_worker_output_ms','isolated_first_path_ms','P_prefill_query_calls','D_prompt_query_calls')
    for split,count in (('calibration',20),('validation',15)):
        native=data('Native',0,split,count)
        for cut in (1,2,4,6,8,12):
            layer=data('Layer',cut,split,count);token=data('Token',cut,split,count)
            if layer is None or token is None:continue
            assert [r['prompt_id'] for r in layer['rows']]==[r['prompt_id'] for r in token['rows']]
            values={'Layer':layer,'Token':token}
            if native is not None:
                assert [r['prompt_id'] for r in native['rows']]==[r['prompt_id'] for r in layer['rows']]
                values['Native']=native
            summary={axis:{field:stats(r[field] for r in value['rows']) for field in fields} for axis,value in values.items()}
            for axis,value in values.items():
                summary[axis]['Hidden_READ_host_ms']=stats(r['host_ms'] for r in value['hidden'])
                summary[axis]['Hidden_bytes']=stats(r['bytes'] for r in value['hidden'])
                summary[axis]['KV_payload_bytes']=stats(r['data']['totalBytes'] for r in value['KV_telemetry'])
                summary[axis]['KV_descriptor_count']=stats(r['data']['descCount'] for r in value['KV_telemetry'])
                summary[axis]['KV_SDK_post_us']=stats(r['data']['postDuration'] for r in value['KV_telemetry'])
            results.append(dict(split=split,N=count,cut=cut,summary=summary,
                Layer_over_Token_isolated_first_path=summary['Layer']['isolated_first_path_ms']['mean']/summary['Token']['isolated_first_path_ms']['mean'],
                Layer_minus_Token_component_mean_ms={field:summary['Layer'][field]['mean']-summary['Token'][field]['mean'] for field in fields[:5]},
                rows=[dict(prompt_id=a['prompt_id'],input_tokens=a['input_tokens'],task=a['task'],Layer=a,Token=b) for a,b in zip(layer['rows'],token['rows'])]))
    value=dict(results=results,evidence='TRACE_ANALYSIS_OF_ISOLATED_SERIAL_MEASUREMENTS',
        scope='One independent start per axis/cut/split. Requests are not server starts. '
        'Host component boundaries partition the observed serial first-output path, not GPU-exclusive time. '
        'SDK bytes include KV block rounding. Hidden transfer is separate. Async first-output service can include a lookahead query. '
        'Different prompt chunk counts and original input lengths must not be replaced by an unchunked microbenchmark ratio. '
        'Online operator-exclusive fractions remain NOT_MEASURED_PRIMARY.')
    write(ROOT/'analysis/BI0_axis_serial_component_comparison_v1.json',value);return value


if __name__=='__main__':main()

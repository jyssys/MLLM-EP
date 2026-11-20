"""Compare identical prompts in serial and loaded runs at host boundaries."""
import json
import numpy as np
from .common import ROOT,write
from .axis_timeline import analyze
from .confirmation_analysis import core_sources


def output_map(name):
    return {row['prompt_id']:row['output_ids'] for row in
            (json.loads(line) for line in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines())}


def shapes(name,role,items):
    rows=json.loads((ROOT/'raw'/name/f'actual_queries_{role}.json').read_text())['rows']
    return {item['prompt_id']:[(q['start'],q['count'],q['padded_queries'],q['requests']) for q in rows
        if q['request_id']==item['request_id'] and (role=='prefill' or q['start']<item['input_tokens'])] for item in items}


def main():
    discovery=ROOT/'analysis/BI0_controlled_fixed_axis_discovery_v3.json'
    if not discovery.exists():return None
    results=[]
    for cell in json.loads(discovery.read_text())['results']:
        online_name=cell['name'];online=analyze(online_name)['rows'];online_by_id={r['prompt_id']:r for r in online}
        online_outputs=output_map(online_name)
        online_shapes={role:shapes(online_name,role,online) for role in ('prefill','decode')}
        for split,count in (('calibration',20),('validation',15)):
            serial_name=f'ContinuationCost_{cell["axis"]}{cell["cut"]}_{split}{count}_BI0_v1'
            receipt=ROOT/'raw'/serial_name/'dispatcher_result.json'
            if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
            serial=analyze(serial_name)['rows'];serial_outputs=output_map(serial_name)
            serial_shapes={role:shapes(serial_name,role,serial) for role in ('prefill','decode')}
            pairs=[]
            for before in serial:
                prompt=before['prompt_id'];after=online_by_id[prompt]
                values={}
                for label,row in (('serial',before),('online',after)):
                    values[label]=dict(P_worker_accept_to_ready_ms=row['P_worker_accept_to_prepare_ms']+row['P_prepare_to_ready_ms'],
                        D_worker_accept_to_first_output_ms=row['D_accept_to_prepare_ms']+row['D_prepare_to_first_worker_output_ms'])
                pairs.append(dict(prompt_id=prompt,input_tokens=before['input_tokens'],values=values,
                    same_prompt_query_shapes=all(serial_shapes[role][prompt]==online_shapes[role][prompt] for role in serial_shapes),
                    same_greedy_output=serial_outputs[prompt]==online_outputs[prompt]))
            summaries={}
            for field in pairs[0]['values']['serial']:
                a=np.asarray([p['values']['serial'][field] for p in pairs]);b=np.asarray([p['values']['online'][field] for p in pairs])
                summaries[field]=dict(serial_mean_ms=float(a.mean()),online_mean_ms=float(b.mean()),
                    online_over_serial_ratio_of_means=float(b.mean()/a.mean()),
                    paired_mean_difference_ms=float(np.mean(b-a)),paired_median_ratio=float(np.median(b/a)))
            results.append(dict(axis=cell['axis'],cut=cell['cut'],split=split,N=count,
                serial_name=serial_name,online_name=online_name,summary=summaries,
                all_prompt_query_shapes_equal=all(p['same_prompt_query_shapes'] for p in pairs),
                greedy_matches=sum(p['same_greedy_output'] for p in pairs),
                core_runtime_source_equal=core_sources(serial_name)==core_sources(online_name),pairs=pairs))
    value=dict(results=results,evidence='TRACE_ANALYSIS',
        scope='One independent serial start and one loaded discovery start per action. Shared prompts remove input-distribution differences. '
        'P service starts at worker acceptance. D service starts at worker acceptance after coordinator admission. '
        'These are host envelopes, not GPU-exclusive time. Serial and online scheduling, ownership and overlap differ. '
        'A difference does not distinguish GPU bandwidth interference from host/control work without another intervention. '
        'This audit does not refit the frozen cost model or use these outcomes in an online controller.')
    write(ROOT/'analysis/BI0_same_prompt_serial_loaded_service_v1.json',value);return value


if __name__=='__main__':main()

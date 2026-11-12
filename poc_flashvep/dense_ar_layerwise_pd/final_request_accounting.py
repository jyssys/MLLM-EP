"""Audit completed work and Host handoff delay without inventing kernel time."""
import json
from .common import *
from .analyze import records,percentile
from .serving_analysis import work


def distribution(values):
    return dict(count=len(values),p50=percentile(values,.5),p95=percentile(values,.95),
        p99=percentile(values,.99),maximum=max(values,default=None))


def main():
    plan=json.loads((ROOT/'state/primed_confirmation_plan_v32.json').read_text());rows=[]
    for group in plan['groups']:
        for label,name in group['arms'].items():
            folder=ROOT/'raw'/name;requests=records(folder/'requests.jsonl');geometry={}
            for role in ('prefill','decode'):
                path=folder/f'measured_geometry_{role}.json'
                if not path.exists():continue
                counts=json.loads(path.read_text())['counts'];totals=dict(prompt_query_rows=0,Decode_query_rows=0,
                    prompt_request_visits=0,Decode_request_visits=0)
                for descriptor,count in counts.items():
                    d=json.loads(descriptor);prefix='prompt' if d['prefilling'] else 'Decode'
                    totals[prefix+'_query_rows']+=count*d['query_tokens']
                    totals[prefix+'_request_visits']+=count*d['requests']
                geometry[role]=totals
            handoffs={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
            delays=[1000*(r['first_token']-handoffs['q'+str(r['sample_id'])]) for r in requests] if handoffs else []
            row=dict(seed=group['seed'],arm=label,name=name,work=work(name),prepared_input_geometry=geometry,
                handoff_Host_ready_to_first_output_ms=distribution(delays),
                scope='ModelRunner prepared-input counts plus completed requests. Handoff delay includes queue, transfer, import, and GPU work. It is not exclusive transfer time or GPU kernel time.')
            if label=='layer':
                row['completed_work_checks']=dict(
                    Source_prompt_rows=geometry['prefill']['prompt_query_rows']==128*8000,
                    D_suffix_rows=geometry['decode']['prompt_query_rows']==128*8000,
                    Source_prompt_visits=geometry['prefill']['prompt_request_visits']==128,
                    D_suffix_visits=geometry['decode']['prompt_request_visits']==128,
                    D_full_Decode_rows=geometry['decode']['Decode_query_rows']==128*15,
                    Hidden_imports=row['work']['actual_hidden_import_count']==128,
                    Hidden_exports=row['work']['actual_hidden_export_count']==128,
                    Hidden_ACK=row['work']['all_hidden_ACKed'])
                assert all(row['completed_work_checks'].values()),row
            rows.append(row)
    write(ROOT/'analysis/final_request_work_and_handoff_accounting.json',dict(rows=rows,
        evidence='MEASURED_HOST_AND_INPUT_GEOMETRY_WITH_STRUCTURAL_LAYER_ACCOUNTING',
        actual_P_saved_kernel_ms='NOT_MEASURED',exclusive_wire_time='NOT_MEASURED'))
    for r in rows:
        if r['arm']=='layer':print(r['seed'],r['handoff_Host_ready_to_first_output_ms'],r['work'])


if __name__=='__main__':main()

"""Audit the current-runtime BI0/BI1 diagnostic without changing evidence."""
import json
from .common import ROOT,write,digest


def main():
    bridge=json.loads((ROOT/'analysis/historical_current_ready_D_comparison_v1.json').read_text())['rows']
    destination=json.loads((ROOT/'analysis/exact_destination_chain_decomposition_v1.json').read_text())['current_historical_workload']
    indexed={(r['axis'],r['BI'],r['seed']):r for r in bridge}
    dindex={(r['axis'],r['BI'],r['seed']):r for r in destination};rows=[]
    for axis in ('Layer','Token'):
        pair={flag:indexed[axis,flag,2026111920] for flag in (False,True)}
        folders={flag:ROOT/'raw'/r['name'] for flag,r in pair.items()}
        configs={flag:json.loads((folder/'configuration.json').read_text()) for flag,folder in folders.items()}
        changes={key:{'BI0':configs[False].get(key),'BI1':configs[True].get(key)}
            for key in set(configs[False])|set(configs[True]) if configs[False].get(key)!=configs[True].get(key)}
        runtime={};queries={}
        for role in ('prefill','decode'):
            actual={flag:json.loads((folder/f'actual_runtime_{role}.json').read_text()) for flag,folder in folders.items()}
            runtime[role]=dict(engine_options_identical=actual[False]['options']==actual[True]['options'],
                UUID_identical=actual[False]['uuid']==actual[True]['uuid'],
                BF16_weights_identical=actual[False]['parameter_dtypes']==actual[True]['parameter_dtypes']==['torch.bfloat16'])
            def shapes(flag):
                values=json.loads((folders[flag]/f'actual_queries_{role}.json').read_text())['rows']
                return [{k:r[k] for k in ('request_id','start','count','requests','padded_queries')} for r in values]
            queries[role]=shapes(False)==shapes(True)
        source={flag:json.loads((folder/'source_hashes.json').read_text()) for flag,folder in folders.items()}
        core=['poc_flashvep/dense_ar_bi0_axis_rootcause/serving.py']+[
            'poc_flashvep/dense_ar_bi0_shortanswer_pd/'+name for name in
            ('standard_worker.py','partial_axis.py','engine_options.py','nixl_transport.py','variable_hidden.py')]
        costs={}
        for flag,row in pair.items():
            d=dindex[axis,flag,2026111920]
            costs['BI1' if flag else 'BI0']=dict(P_submit_to_ready_ms=row['mean_stages_ms']['P_submit_to_parent_ready_ms'],
                D_submit_to_first_worker_ms=d['mean_D_preparation_ms'],
                subsequent_15_token_worker_residency_ms=d['mean_decode_residency_ms'],
                mean_worker_emission_interval_ms=d['mean_decode_residency_ms']/15)
        rows.append(dict(axis=axis,config_differences=changes,
            configs_SHA256={str(flag):digest(folder/'configuration.json') for flag,folder in folders.items()},
            actual_runtime=runtime,actual_query_geometry_equal=queries,
            critical_source_equal={path:source[False][path]==source[True][path] for path in core},host_costs=costs))
    write(ROOT/'analysis/current_runtime_numerical_bridge_contract.json',dict(rows=rows,evidence='DIAGNOSTIC',
        N=72,input=8000,output=16,QPS=6,seed=2026111920,starts_per_axis_mode=1,
        scope='Current runtime only. Config differences include BI and dispatcher eligibility/reporting metadata. '
        'Actual engine options,query geometry and critical source equality are audited separately. '
        'Worker emission residency is not GPU-exclusive kernel time. No start-level CI,BI0/BI1 strict-output equality or BI1 Native-relative safety claim. '
        'This does not isolate every Historical-to-Current runtime change.'))
    print(json.dumps([dict(axis=r['axis'],host_costs=r['host_costs'],config_difference_keys=list(r['config_differences'])) for r in rows],indent=2))


if __name__=='__main__':main()

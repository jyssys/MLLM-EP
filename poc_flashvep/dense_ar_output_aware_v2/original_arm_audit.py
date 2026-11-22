"""Verify both actual original-model replicas and their complete query coverage."""
import collections
import json
from .common import ROOT,write,digest

def audit_agg(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text());assert cfg['method']=='Agg'
    assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    requests={f'q{x["sample_id"]}':x for x in map(json.loads,(folder/'requests.jsonl').read_text().splitlines())}
    expected={f'q{i}' for i in range(len(data))};replicas=[];seen=set()
    for replica in ('replica0','replica1'):
        runtime=json.loads((folder/f'actual_runtime_{replica}.json').read_text())
        assert runtime['model_layers']==36 and runtime['model_forward_module']=='vllm.model_executor.models.qwen3'
        assert runtime['compile_mode']=='3' and runtime['scheduler_B']==8192 and runtime['scheduler_S']==1024 and runtime['chunked_prefill']
        query=json.loads((folder/f'actual_queries_{replica}.json').read_text())['rows']
        states=json.loads((folder/f'prompt_state_{replica}.json').read_text())['requests']
        scheduler=json.loads((folder/f'scheduler_queues_{replica}.json').read_text())['rows']
        exact_path=folder/f'scheduler_preemptions_{replica}.json'
        exact=json.loads(exact_path.read_text())['rows'] if exact_path.exists() else None
        groups=collections.defaultdict(list)
        for row in query:
            if row['request_id'] in expected:groups[row['request_id']].append(row)
        assert groups and not seen.intersection(groups)
        role_number=int(replica[-1]);assert all(int(rid[1:])%2==role_number for rid in groups)
        observations=[]
        for rid,values in groups.items():
            index=int(rid[1:]);prompt=data[index];state=states[rid]
            assert state['query_coverage_end']==prompt['input_tokens'] and len(state['KV'])==36 and state['final_hidden_SHA256']
            assert [x['layer'] for x in state['KV']]==list(range(36))
            assert all(x['bytes']==prompt['input_tokens']*4096 for x in state['KV'])
            end=0;last_time=None;resets=[];recomputed=0
            for row in sorted(values,key=lambda r:r['time']):
                if row['start']!=end:
                    assert row['start']==0 and end>0,('Unexplained query gap or nonzero restart',rid,end,row)
                    # A reset is not accepted merely because a query repeated.
                    # Require the original scheduler's sampled PREEMPTED state
                    # in the same recorded interval. Exact per-ID preemption
                    # events are added prospectively in the final v2 source.
                    witness=[s for s in scheduler if (last_time-.05)<=s['time']<=row['time']+.005
                        and s['waiting_status'].get('PREEMPTED',0)>0] if exact is None else [s for s in exact
                        if s['request_id'].split('-')[0]==rid and s['computed_before']==end
                        and last_time-.05<=s['time']<=row['time']+.005]
                    assert witness,('Unexplained duplicate work without a scheduler-preemption witness',rid,end,row)
                    resets.append(dict(time=row['time'],previous_computed_end=end,
                        scheduler_witness_time=witness[-1]['time'],exact_per_ID_witness=exact is not None,evidence='TRACE_ANALYSIS'))
                    recomputed+=end;end=0
                end+=row['count']
                last_time=row['time']
            required=prompt['input_tokens']+requests[rid]['output_tokens']-1
            assert end>=required
            observations.append(dict(request_id=rid,valid_prompt_tokens=prompt['input_tokens'],actual_forward_queries=end,
                minimum_required_queries=required,async_terminal_extra_queries=end-required))
            observations[-1].update(recomputed_query_rows=recomputed,preemption_compatible_resets=resets)
        seen.update(groups);replicas.append(dict(role=replica,requests=len(groups),observations=observations,
            actual_model_layers=36,compile_mode=3,original_model_forward=True,
            recomputed_query_rows=sum(r['recomputed_query_rows'] for r in observations),exact_preemption_events_available=exact is not None))
    assert seen==expected
    result=dict(passed=True,name=name,n=len(data),replicas=replicas,dataset_SHA256=cfg['dataset_SHA256'],
        evidence='CORRECTNESS_ONLY',scope='Actual original-model query coverage,all valid prompt-KV planes and final-hidden hashes are observed. '
            'No cross-batch bitwise equality is claimed. Task quality is a separate gate. '
            'Original-scheduler preemption-compatible recomputation and extra async terminal work are counted,not hidden. '
            'Exact per-ID preemption events are required when the source provides them. Older sources expose only sampled PREEMPTED witnesses. '
            'First-completion KV hashes do not prove later recomputation bytes are identical.')
    write(ROOT/'analysis'/(name+'_original_model_state_audit.json'),result);return result

if __name__=='__main__':
    import sys
    value=audit_agg(sys.argv[1]);print('ORIGINAL_AGG_BOTH_REPLICAS_FULL_STATE_PASS',[(r['role'],r['requests']) for r in value['replicas']])

"""Use actual completed Dynamic work to define declared discovery controls."""
import collections,random,fcntl
from .common import *
from .configuration import base
from .method_discovery_v2 import method_settings
from .run import execute
from .qualification import against_previous
from .analyze import metrics,records


def run(arm,name,**changes):
    options=method_settings()
    options.update(async_decode=True,separate_nominal_objective=True,cost_model='coalesced_cost_model_v2.json',
        phase='ASYNC_DECODE_AND_MATCHED_BUDGET_DISCOVERY')
    cfg=base(name,arm,count=128,cap=16,rate=4.8,**dict(options,**changes))
    assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
    value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
    actual=[r.get('completed_D_prefill_layers',0) for r in records(ROOT/'raw'/name/'requests.jsonl')]
    print('ASYNC_MATCHED',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],sum(actual),flush=True)
    return actual


def spread(blocks,seed):
    values=list(blocks);random.Random(seed).shuffle(values)
    return [x for group in values for x in group]


def main():
    assert json.loads((ROOT/'analysis/async_D_online_work_gate.json').read_text())['passed']
    try:
        name='async_D_DIRECT_nominal_N128_O16_q4.8_v1';actual=run('DIRECT',name)
        hist=collections.Counter(actual);migrated=sum(x>0 for x in actual);work=sum(actual)
        assert work>0
        # The histogram is selection data, not hidden confirmation data.
        # Keep same-cut runs broad enough to permit legal compatible batching.
        blocks=[]
        for layer in sorted(hist):
            left=hist[layer]
            while left:
                n=min(8,left);blocks.append([layer]*n);left-=n
        histogram=spread(blocks,6141)
        fraction=spread([[12]*min(8,migrated-i) for i in range(0,migrated,8)]+
            [[0]*min(8,128-migrated-i) for i in range(0,128-migrated,8)],6142)
        full,remainder=divmod(work,12);budget=[12]*full+([remainder] if remainder else [])
        assert not remainder or remainder in (2,4,8)
        budget+=([0]*(128-len(budget)));budget_blocks=[]
        for layer in sorted(set(budget)):
            left=budget.count(layer)
            while left:
                n=min(8,left);budget_blocks.append([layer]*n);left-=n
        budget=spread(budget_blocks,6143)
        controls={'same_request_fraction':fraction,'same_layer_work':budget,'same_cut_histogram':histogram}
        results={}
        for label,assignment in controls.items():
            assert len(assignment)==128
            observed=run('STATIC',f'async_D_STATIC_matched_{label}_N128_O16_q4.8_v1',
                static_layers=12,migration_fraction=1.,static_layers_by_request=assignment)
            results[label]=dict(actual_requests=sum(x>0 for x in observed),actual_layers=sum(observed),
                actual_histogram=dict(collections.Counter(observed)),requested_histogram=dict(collections.Counter(assignment)),
                assigned_work_completed_exactly=collections.Counter(observed)==collections.Counter(assignment))
        write(ROOT/'analysis/async_D_discovery_matched_controls.json',dict(dynamic=name,
            dynamic_requests=migrated,dynamic_layers=work,dynamic_histogram=dict(hist),controls=results,
            evidence='MEASURED',scope='Discovery-only controls use the completed discovery histogram. They do not consult hidden evaluation decisions. Source and destination limits match, but realized group geometry can differ and must be audited. Matching totals alone does not establish an independent policy gain.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()

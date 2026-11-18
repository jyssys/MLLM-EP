"""Run the user-priority real Last2/Last4 continuation checks as one lease."""
import collections
import contextlib
import fcntl
import json
import statistics
import time
from . import common
from .common import ROOT,write,save
from .configuration import base,matched_token_plan
from .prepare import qualification
from .qualification import gate
from .run import execute
from .analysis import metrics
from .host_paths import records
from ..dense_ar_slo_layer_pd.analyze import percentile


def configuration(method,layers,qualify=False,probe=False):
    stage='qual' if qualify else 'projection_probe' if probe else 'actual_forward'
    name=f'priority_suffix_{stage}_{method}_Last{layers}_N18_O16_v1'
    requested=4 if method=='Token' and layers==2 else layers
    if qualify:
        cfg=qualification(method,18,requested,version=111)
    else:
        cfg=base(name,method,requested,18,16,.5,2026111802,'heldout128_8000.json',
            phase='PRIORITY_ACTUAL_SUFFIX_SERIAL_DIAGNOSTIC',D_admission_batch=1,
            execution_events=True,boundary_events=False,clean_timing=False,
            arrival_offsets=[2.*i for i in range(18)],warmup_count=2,
            entry_module='poc_flashvep.dense_ar_matched_axis_pd.priority_suffix_serving')
    cfg.update(name=name,static_layers=layers,D_admission_batch=1,
        reference_name='HISTORICAL_original_outputs',matrix_stage='priority_actual_suffix',
        common_prospective_KV_reservation=True,execution_contract_version=3,
        priority_projection_events=probe,
        witness_hidden_partition_tokens=[432,448,880,896],axis_state_save_tensors=False)
    if method=='Token':
        cfg.update(token_suffix=448 if layers==2 else 896,
            token_suffix_by_request=matched_token_plan(layers,18,group_size=1),layer_cut=36-layers)
    return cfg


def forward_rows(name):
    folder=ROOT/'raw'/name
    value=json.loads((folder/'CUDA_event_enclosures_decode.json').read_text())
    rows=[r for r in value['rows'] if r['prefilling'] and all(
        rid.startswith('q') and rid[1:].isdigit() and int(rid[1:])>=9 for rid in r['requests'])]
    assert len(rows)==9 and all(len(r['requests'])==1 for r in rows),(name,len(rows))
    assert sorted(r['requests'][0] for r in rows)==sorted('q'+str(i) for i in range(9,18))
    return rows


def summarize(names):
    rows=[]
    for name in names:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        observed=forward_rows(name)
        values=[r['elapsed_ms'] for r in observed]
        requests=[r for r in records(ROOT/'raw'/name/'requests.jsonl') if r['sample_id']>=9]
        result=dict(name=name,method=cfg['evaluation_arm'],layers=cfg['static_layers'],
            projection_probe=cfg['priority_projection_events'],sampled_requests=9,
            D_actual_forward_event_ms=dict(p50=percentile(values,.5),p95=percentile(values,.95),mean=statistics.mean(values),samples=values),
            serial_client_TTFT_ms=dict(p50=percentile([r['ttft_ms'] for r in requests],.5),
                p95=percentile([r['ttft_ms'] for r in requests],.95)),
            actual_D_query_rows=[r['query_tokens'] for r in observed],
            evidence='MEASURED',scope='Actual one-request transferred-state continuation. First9 real requests warm both adjacent suffix geometries. Last9 form an exact matched-row period. One process start per arm. No independent-start confidence interval. CUDA event enclosures are not exclusive kernel time. Serial client TTFT is not online queueing performance.')
        if cfg['priority_projection_events']:
            operators=json.loads((ROOT/'raw'/name/'actual_projection_events_decode.json').read_text())['rows']
            groups=collections.defaultdict(list)
            for row in operators:
                assert len(row['requests'])==1
                groups[row['requests'][0]].append(row)
            expected=cfg['static_layers'] if cfg['evaluation_arm']=='Layer' else 36
            first=36-expected if cfg['evaluation_arm']=='Layer' else 0
            assert len(groups)==9
            for values_by_request in groups.values():
                assert len(values_by_request)==expected*4
                visits=collections.Counter(int(r['module'].split('.layers.')[1].split('.')[0]) for r in values_by_request)
                assert visits==collections.Counter({i:4 for i in range(first,36)}),visits
            sums=[sum(r['elapsed_ms'] for r in group) for group in groups.values()]
            result['actual_projection_event_sums_ms']=dict(p50=percentile(sums,.5),p95=percentile(sums,.95),mean=statistics.mean(sums),samples=sums,
                observed_projection_calls=len(operators),executed_layers=list(range(first,36)),
                scope='Sum events around the actual projections in every executed layer. No repeated representative-layer estimate. Not an exclusive kernel-duration sum.')
        rows.append(result)
    comparisons=[]
    for layers in (2,4):
        a=next(r for r in rows if r['layers']==layers and r['method']=='Layer' and not r['projection_probe'])
        b=next(r for r in rows if r['layers']==layers and r['method']=='Token' and not r['projection_probe'])
        comparisons.append(dict(layers=layers,Layer=a['name'],Token=b['name'],
            Token_over_Layer_actual_forward_p50=b['D_actual_forward_event_ms']['p50']/a['D_actual_forward_event_ms']['p50']))
    for row in rows:
        if row['projection_probe']:
            control=next(r for r in rows if r['layers']==row['layers'] and r['method']==row['method'] and not r['projection_probe'])
            ratio=row['D_actual_forward_event_ms']['p50']/control['D_actual_forward_event_ms']['p50']
            row['whole_forward_probe_over_control']=ratio
            row['whole_forward_probe_5pct_screen']=abs(ratio-1)<=.05
            row['evidence']='DIAGNOSTIC'
    write(ROOT/'analysis/priority_actual_suffix_results.json',dict(time=time.time(),rows=rows,comparisons=comparisons,
        scope='Real Last2/Last4 P→D execution with original BF16 Qwen3-8B. Equal Transformer token-layer rows across a nine-request period. No TTFT or throughput improvement under the primary QPS sweep is inferred. Projection instrumentation remains a diagnostic observer with its own interference screen.'))
    print('PRIORITY_ACTUAL_SUFFIX_RESULTS',json.dumps(comparisons),flush=True)


def main():
    common.initialize()
    configs=[configuration('Native',0,True),configuration('Layer',2,True),configuration('Token',2,True)]
    timing=[configuration(method,layers,False,probe) for probe in (False,True)
        for method,layers in (('Layer',4),('Token',4),('Token',2),('Layer',2))]
    for cfg in configs+timing:save(cfg)
    write(ROOT/'state/priority_actual_suffix_design.json',dict(frozen_epoch=time.time(),names=[c['name'] for c in configs+timing],
        primary_continues_unchanged=True,B=8192,S=1024,admission=1,count=18,warm_period=9,timed_period=9,
        suffix_histograms={str(m):dict(collections.Counter(matched_token_plan(m,18,group_size=1))) for m in (2,4)},
        scope='User-priority actual execution check. Hold the existing quiet lease across this bounded batch after the running job finishes. Other campaign controllers remain queued. No active GPU job is terminated. The original primary configuration and frozen inference files are unchanged.'))
    original=common.QUIET_LOCK
    with original.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        class BorrowedLease:
            def open(self,*args,**kwargs):return contextlib.nullcontext(lease)
        common.QUIET_LOCK=BorrowedLease()
        names=[]
        try:
            reference=configs[0]['name']
            for index,cfg in enumerate(configs+timing):
                write(ROOT/'state/priority_actual_suffix_progress.json',dict(time=time.time(),index=index,total=len(configs+timing),name=cfg['name'],status='RUNNING'))
                assert execute(cfg['name']),cfg['name']
                if cfg['correctness_only']:
                    gate(cfg['name'],reference)
                else:
                    metrics(cfg['name']);names.append(cfg['name'])
                    if len(names)>=4:summarize(names)
            write(ROOT/'state/priority_actual_suffix_complete.json',dict(time=time.time(),names=names,passed=True))
        finally:
            common.QUIET_LOCK=original


if __name__=='__main__':
    main()

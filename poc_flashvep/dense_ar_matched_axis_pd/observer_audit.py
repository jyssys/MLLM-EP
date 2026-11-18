"""Explain rejected Nsight runs without rewriting their dispatcher receipts."""
import json
import statistics
from .common import ROOT,write
from .host_paths import records
from .analysis import output_gate
from ..dense_ar_slo_layer_pd.analyze import percentile


def summarize(name):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    exits=json.loads((folder/'worker_exits.json').read_text())
    assert status['status']=='COMPLETE' and exits['passed']
    rows=records(folder/'requests.jsonl')
    assert len(rows)==cfg['count'] and output_gate(name)['passed']
    window=json.loads((folder/'window.json').read_text())
    intervals=[1000*(b['time']-a['time']) for r in rows
        for a,b in zip(r['token_events'],r['token_events'][1:])]
    values=dict(TTFT_p95_ms=percentile([r['ttft_ms'] for r in rows],.95),
        mean_TBT_ms=statistics.mean(intervals),ITL_p99_ms=percentile(intervals,.99),
        TPOT_p95_ms=percentile([r['tpot_ms'] for r in rows],.95),
        maximum_ITL_ms=max(intervals),
        output_tokens_s=sum(r['output_tokens'] for r in rows)/(window['end']-window['start']))
    audits=[json.loads(p.read_text()) for p in folder.glob('effective_runtime_*.json')]
    assert len(audits)==2 and all(r['actual_scheduler_max_num_batched_tokens']==8192 and
        r['actual_scheduler_max_num_seqs']==1024 for r in audits)
    roles=('replica0','replica1') if cfg['arm']=='DP2' else ('prefill','decode')
    ready={}
    for role,gpu in zip(roles,cfg['gpus']):
        path=folder/(role+'_ready.json')
        if path.exists():ready[(gpu,json.loads(path.read_text())['pid'])]=role
    foreign={(r['GPU'],r['pid']) for p in folder.glob('NVML-summary-*.json')
        for r in json.loads(p.read_text()).get('foreign_process_observations',[])}
    receipt=json.loads((folder/'dispatcher_result.json').read_text())
    return dict(name=name,metrics=values,original_dispatcher=receipt,
        foreign_GPU_PID_pairs=sorted(foreign),ready_GPU_PID_pairs=sorted(ready),
        foreign_PIDs_match_trusted_worker_ready_messages=bool(foreign) and foreign<=set(ready),
        actual_BS_passed=True,output_passed=True,evidence='DIAGNOSTIC',
        scope='Post-hoc inspection only. Nsight can start workers in separate sessions. Match their GPU/PID to the trusted worker-ready channel. A full live parent chain was not recorded. Do not change the original isolation verdict or promote this start to primary clean performance.')


def main():
    path=ROOT/'analysis/online_profiler_observer_gates.json'
    if not path.exists():return
    rows=[]
    for gate in json.loads(path.read_text()):
        try:
            before,profiled,after=[summarize(name) for name in gate['names']]
            tests={}
            for key in before['metrics']:
                a,b,c=[r['metrics'][key] for r in (before,profiled,after)]
                control=(a+c)/2
                tests[key]=dict(clean_before=a,profiled=b,clean_after=c,
                    profiled_over_clean_controls=b/control,controls_after_over_before=c/a,
                    within_5pct=abs(b/control-1)<=.05 and abs(c/a-1)<=.05)
            rows.append(dict(method=gate['method'],cap=gate['cap'],names=gate['names'],
                controls=[before,profiled,after],tests=tests,
                original_observer_gate=gate['passed'],
                clean_timing_qualified=False,evidence='DIAGNOSTIC',
                scope='Original dispatcher/isolation failure is preserved. These ratios quantify observer effects. No primary serving or physical-idle claim follows from this audit.'))
        except BaseException as error:
            rows.append(dict(method=gate['method'],cap=gate['cap'],error=repr(error),evidence='FAILED'))
    write(ROOT/'analysis/online_profiler_secondary_audit.json',dict(rows=rows,
        scope='No raw receipt is overwritten. Post-hoc worker identity matching does not silently relax the ownership guard.'))


if __name__=='__main__':
    main()

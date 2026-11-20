"""Keep coordinator waiting and worker scheduler waiting distinct."""
import json
import numpy as np
from .common import ROOT,write
from .serving_analysis import time_weighted


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    window=json.loads((folder/'window.json').read_text())
    rows=[json.loads(line) for line in (folder/'coordinator_queues.jsonl').read_text().splitlines()]
    keys=('P_pipeline_waiting','D_pipeline_waiting','P_pipeline_active','D_pipeline_active')
    queues={key:time_weighted([r for r in rows if r.get(key) is not None],key,window) for key in keys}
    blocking=dict(status='NOT_INSTRUMENTED')
    if cfg.get('controlled_pipeline'):
        starts=sorted(json.loads(line)['time'] for line in (folder/'axis_decisions.jsonl').read_text().splitlines())
        firsts=sorted(json.loads(line)['first_token'] for line in (folder/'requests.jsonl').read_text().splitlines())
        durations={key:0. for key in ('P_active_host_phase','P_backlog_source_credit_full',
            'P_backlog_other_host_gap','P_no_pending_request')}
        maximum_owned=0;covered=0.
        for index,row in enumerate(rows):
            left=max(window['start'],row['time']);right=min(window['end'],rows[index+1]['time'] if index+1<len(rows) else window['end'])
            if right<=left:continue
            owned=int(np.searchsorted(starts,row['time'],side='right')-np.searchsorted(firsts,row['time'],side='right'))
            assert 0<=owned<=cfg.get('source_credit_limit',4),(name,owned)
            maximum_owned=max(maximum_owned,owned)
            if row['P_pipeline_active']:key='P_active_host_phase'
            elif not row['P_pipeline_waiting']:key='P_no_pending_request'
            elif owned>=cfg.get('source_credit_limit',4):key='P_backlog_source_credit_full'
            else:key='P_backlog_other_host_gap'
            durations[key]+=right-left;covered+=right-left
        blocking=dict(covered_seconds=covered,seconds=durations,
            fractions={key:value/covered for key,value in durations.items()},maximum_reconstructed_source_owned=maximum_owned,
            scope='Source ownership is reconstructed from dispatch and first client output. '
            'These are sampled host admission states, not P GPU busy/idle intervals. '
            'Credit-full blocking can reflect downstream D service. It is not proof of P compute saturation.')
    arrival_duration=window['last_arrival']-window['start'];early_end=window['start']+.25*arrival_duration
    late_start=window['start']+.75*arrival_duration
    growth={}
    for key in keys[:2]:
        early=[r[key] for r in rows if r.get(key) is not None and r['time']<=early_end]
        late=[r[key] for r in rows if r.get(key) is not None and late_start<=r['time']<=window['last_arrival']]
        growth[key]=dict(first_quarter_mean=float(np.mean(early)),last_quarter_mean=float(np.mean(late)),
            difference=float(np.mean(late)-np.mean(early))) if early and late else dict(status='NOT_INSTRUMENTED')
    return dict(name=name,method=cfg['method'],cut=cfg.get('axis_cut',0),QPS=cfg['offered_QPS'],
        seed=cfg['arrival_seed'],controlled=cfg.get('controlled_pipeline',False),queues=queues,
        sampled_arrival_window_growth=growth,source_admission_states=blocking,
        scope='Coordinator queues sampled approximately every50ms. D_pipeline_waiting contains completed P work awaiting D admission. '
        'It is not the D worker decode-ready queue, a transfer duration or a physical GPU-idle measurement. '
        'Arrival-window growth is descriptive. A finite cohort does not establish stable long-run capacity.')


def main():
    patterns=('ControlledDiscovery_*_BI0_v3_metrics.json','Confirmation_*_BI0_v2_metrics.json','FixedQPS_*_BI0_v1_metrics.json','RootCut12_*_BI0_v1_metrics.json','LowQPSControl_*_BI0_v1_metrics.json')
    values=[]
    for file in sorted({p for pattern in patterns for p in (ROOT/'analysis').glob(pattern)}):
        name=json.loads(file.read_text())['name']
        if (ROOT/'raw'/name/'coordinator_queues.jsonl').exists():values.append(analyze(name))
    result=dict(results=values,evidence='TRACE_ANALYSIS',scope='Sampled host states. Not GPU kernel activity or physical decode-safe windows.')
    write(ROOT/'analysis/BI0_PD_host_queue_summaries_v1.json',result);return result


if __name__=='__main__':main()

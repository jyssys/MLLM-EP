"""Analyze event brackets without relabeling uncovered time as physical idle."""
import csv
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .profile_analysis import union,complement,intersection,duration,windows

NAME='ec_GPU_ENCLOSURES_C19_CPC_q2_v0'


def save(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    import argparse
    global NAME
    parser=argparse.ArgumentParser();parser.add_argument('--name',default=NAME)
    args=parser.parse_args();assert args.name.isidentifier() or all(c.isalnum() or c in '._' for c in args.name)
    NAME=args.name
    preserve()
    qualification=ROOT/'analysis'/f'{NAME}_qualification.json'
    observer=json.loads(qualification.read_text())
    run=NAME+'_profile'
    done=json.loads((ROOT/'state'/f'{run}_complete.json').read_text())
    requests=[json.loads(line) for line in (ROOT/'raw'/f'{run}_requests.jsonl').read_text().splitlines()]
    left=done['monotonic_start'];right=left+done['duration_seconds']
    arrivals=[r['scheduled_arrival'] for r in requests]
    roi={'full_cohort':(left,right),'central_arrival_window':tuple(np.percentile(arrivals,[10,90]))}
    stages={};summary=[];window_rows=[];overlaps=[]
    for role in ['encoder','prefill','decode']:
        folder=ROOT/'raw/gpu_enclosures'/NAME/role
        anchors=json.loads((folder/'anchors.json').read_text())
        with (folder/'forward_enclosures.csv').open() as stream:data=list(csv.DictReader(stream))
        assert len(data)==anchors['events']>0
        intervals=np.asarray([[float(r['begin_ns'])/1e9,float(r['end_ns'])/1e9] for r in data])
        states=json.loads((folder/'scheduler.json').read_text())['rows']
        stages[role]=dict(intervals=intervals,anchors=anchors,source_SHA256=digest(folder/'forward_enclosures.csv'))
        for label,(lo,hi) in roi.items():
            enclosed=union(intervals,lo,hi);uncovered=complement(enclosed,lo,hi)
            observed=[r for r in states if lo<=r['time_ns']/1e9<=hi]
            summary.append(dict(ROI=label,role=role,duration_seconds=hi-lo,
                execute_bracket_fraction=duration(enclosed)/(hi-lo),
                outside_execute_brackets_fraction=duration(uncovered)/(hi-lo),
                mean_observed_waiting=float(np.mean([r['waiting'] for r in observed])) if observed else None,
                maximum_observed_waiting=max((r['waiting'] for r in observed),default=None),
                maximum_scheduled_requests=max(int(r['scheduled_requests']) for r in data),
                maximum_scheduled_tokens=max(int(r['scheduled_tokens']) for r in data),
                endpoint_alignment_uncertainty_max_us=max(anchors[k]['uncertainty_ns'] for k in ['left','right'])/1000,
                observer_gate_passed=observer['observer_gate_passed'],evidence='CUDA_EVENT_EXECUTE_BRACKETS'))
            for lo2,hi2 in uncovered:
                window_rows.append(dict(ROI=label,role=role,begin=lo2,end=hi2,duration_ms=(hi2-lo2)*1000,
                    meaning='Outside model-runner execute brackets, not physical idle or safe slack'))
    results={}
    for label,(lo,hi) in roi.items():
        enclosed={r:union(v['intervals'],lo,hi) for r,v in stages.items()}
        uncovered={r:complement(enclosed[r],lo,hi) for r in stages}
        simultaneous=intersection(uncovered['encoder'],uncovered['decode'])
        during_p=intersection(simultaneous,enclosed['prefill'])
        p_span=duration(enclosed['prefill'])
        results[label]=dict(stage_gap_statistics={r:windows(v) for r,v in uncovered.items()},
            E_D_outside_brackets_fraction=duration(simultaneous)/(hi-lo),
            E_D_outside_brackets_during_P_bracket_full_fraction=duration(during_p)/(hi-lo),
            E_D_outside_brackets_during_P_bracket_P_fraction=duration(during_p)/p_span if p_span else None)
        overlaps.append(dict(ROI=label,**{k:v for k,v in results[label].items() if k!='stage_gap_statistics'},
            evidence='Execution-bracket overlap, not physical idle or actionable slack'))
    output=ROOT/'analysis'/f'{NAME}_enclosure_analysis_v0.json'
    assert not output.exists()
    qps=json.loads((ROOT/'configs'/f'{NAME}.json').read_text())['QPS']
    write(output,dict(time=time.time(),QPS=qps,observer_gate_passed=observer['observer_gate_passed'],
        qualification_SHA256=digest(qualification),rows=summary,ROIs=results,
        actual_extra_prefill='NOT_RUN',kernel_busy='NOT_MEASURED_BY_THIS_OBSERVER',
        safe_headroom='NOT_CONFIRMED',task_quality='PENDING',
        limitations=['CUDA events bracket model-runner execution, including host submission gaps and dependencies.',
            'Export work and unrelated GPU streams can occur outside these brackets.',
            'Outside-bracket intervals are not physical kernel idle or safe executable windows.',
            'Scheduler samples are not an exact time-weighted queue trajectory.',
            'The central arrival window is not proof of sustainable steady state.']))
    save(ROOT/'analysis'/f'{NAME}_enclosure_summary_v0.csv',summary)
    save(ROOT/'analysis'/f'{NAME}_outside_bracket_windows_v0.csv',window_rows)
    save(ROOT/'analysis'/f'{NAME}_overlap_v0.csv',overlaps)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));cdf=[]
    for role in ['encoder','decode']:
        values=sorted(r['duration_ms'] for r in window_rows if r['ROI']=='central_arrival_window' and r['role']==role)
        if not values:continue
        probability=np.arange(1,len(values)+1)/len(values)
        ax.step(values,probability,where='post',label=role)
        cdf.extend(dict(role=role,duration_ms=float(v),CDF=float(p)) for v,p in zip(values,probability))
    ax.set_xscale('log');ax.set_xlabel('Outside execute brackets (ms), not physical idle')
    ax.set_ylabel('Window-count CDF');ax.grid(alpha=.2);ax.legend()
    ax.set_title(f'CUDA-event brackets | QPS{qps:g} Main100 | Observer gate: '+str(observer['observer_gate_passed']))
    fig.text(.01,.01,'Unrelated streams/transfer may remain active. No extra prefill or safe-headroom claim.',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1])
    stem=ROOT/'figures'/f'{NAME}_outside_bracket_CDF_v0'
    for suffix in ['png','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=170)
    save(stem.with_suffix('.csv'),cdf);plt.close(fig)
    preserve();print('CUDA_EXECUTE_BRACKET_ANALYSIS',observer['observer_gate_passed'],results,flush=True)


if __name__=='__main__':main()

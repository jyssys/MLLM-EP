"""Clip both GPU roles to one host-time window in each qualified capture."""
import csv
import json
import time
import numpy as np
from .common import ROOT,write
from .window_analysis import merge,duration,queues,distribution
from .confirmed_figures import mean_ci


def anchor(roi,side):
    row=roi[side]
    return (row['host_lower_ns']+row['host_upper_ns'])//2


def map_cupti(roi,host):
    left,right=anchor(roi,'left'),anchor(roi,'right')
    assert right>left
    return roi['left']['cupti_ns']+(host-left)*(roi['right']['cupti_ns']-roi['left']['cupti_ns'])//(right-left)


def main():
    cells=[]
    for qps in (1.8,2.8):
        path=ROOT/'analysis'/f'recovered_native_activity_q{qps}_v1.json'
        if not path.exists():
            continue
        source=json.loads(path.read_text())
        if source.get('passed') is not True:
            status='DIAGNOSTIC_FAILED_OBSERVER' if source.get('passed') is False else 'PENDING_OBSERVER'
            cells.append(dict(QPS=qps,status=status,primary_data=None))
            continue
        results=[]
        for item in source['results']:
            name=item['names'][1]
            folder=ROOT/'raw'/name
            for measured in item['names']:
                score=json.loads((ROOT/'analysis'/(measured+'_quality_gate.json')).read_text())
                assert score['passed'],measured
            rois={role:json.loads((folder/f'activity_{role}/actual_ROI.json').read_text())
                for role in ('prefill','decode')}
            left=max(r['left']['host_upper_ns'] for r in rois.values())
            right=min(r['right']['host_lower_ns'] for r in rois.values())
            assert right>left
            roles={}
            for role,roi in rois.items():
                a,b=map_cupti(roi,left),map_cupti(roi,right)
                kernels,copies=[],[]
                with (folder/f'activity_{role}/kernels.csv').open() as stream:
                    for row in csv.DictReader(stream):
                        span=max(a,int(row['start_ns'])),min(b,int(row['end_ns']))
                        if span[1]<=span[0]:
                            continue
                        if row['kind']=='kernel':
                            kernels.append(span)
                        elif row['kind'] in ('memcpy','memcpy2'):
                            copies.append(span)
                state=json.loads((folder/f'scheduler_queues_{role}.json').read_text())['rows']
                q=queues(state,left/1e9,right/1e9,('waiting_total','running','active_decode',
                    'decode_ready_waiting','decode_ready_including_first_query','remote_KV_waiting','free_KV_blocks'))
                gaps=[]
                cursor=a
                for kl,kr in merge(kernels):
                    if kl>cursor:gaps.append((cursor,kl))
                    cursor=max(cursor,kr)
                if cursor<b:gaps.append((cursor,b))
                interior=[(gl,gr) for gl,gr in gaps if gl>a and gr<b]
                gap_ms=[(gr-gl)/1e6 for gl,gr in interior]
                roles[role]=dict(kernel_busy_fraction=duration(kernels)/(b-a),
                    kernel_inactivity_fraction=1-duration(kernels)/(b-a),
                    interior_observed_kernel_gaps_ms=gap_ms,
                    interior_kernel_gap_distribution_ms=distribution(gap_ms),
                    boundary_censored_gap_count=len(gaps)-len(interior),
                    observed_copy_union_fraction=duration(copies)/(b-a),
                    kernel_or_observed_copy_busy_fraction=duration(kernels+copies)/(b-a),queue=q,
                    maximum_anchor_uncertainty_ns=max(roi[k]['host_upper_ns']-roi[k]['host_lower_ns'] for k in ('left','right')),
                    full_GPU_physical_idle='NOT_ESTABLISHED',decode_safe_future_budget='NOT_IDENTIFIED')
            results.append(dict(name=name,shared_host_left_ns=left,shared_host_right_ns=right,
                shared_window_seconds=(right-left)/1e9,roles=roles))
        assert len(results)==3
        summaries={}
        for role in ('prefill','decode'):
            summaries[role]={}
            for field in ('kernel_busy_fraction','kernel_inactivity_fraction'):
                value,ci=mean_ci([r['roles'][role][field] for r in results])
                summaries[role][field]=dict(mean=value,CI95=ci,independent_profile_starts=3)
            summaries[role]['queue_start_means']={field:float(np.mean([r['roles'][role]['queue']['time_weighted_mean'][field]
                for r in results])) for field in results[0]['roles'][role]['queue']['time_weighted_mean']}
        cells.append(dict(QPS=qps,status='MEASURED_KERNEL_ACTIVITY',results=results,summaries=summaries))
    write(ROOT/'analysis/recovered_native_shared_window_activity_v1.json',dict(time=time.time(),cells=cells,
        evidence='MEASURED_KERNEL_ACTIVITY_WHERE_OBSERVER_PASSED',
        scope='Each paired capture uses the same conservative common host-time intersection for P and D. '
            'Two-anchor clock mapping accounts for measured offset drift. Anchor uncertainty remains recorded. '
            'Kernel busy is the union of observed kernel intervals. Copy unions are separate and can overlap kernels. '
            'Three profiles come from three independent Clean/Profile/Clean triplets whose full/exposed metric CIs pass5percent. '
            'This does not establish all-engine physical idle,a future executable window,or sustainable capacity. '
            'The common READ/compute policy is not the original motivation runtime.'))
    print([(c['QPS'],c['status'],c.get('summaries')) for c in cells],flush=True)


if __name__=='__main__':
    main()

"""Check instrumentation parity and service-clock overhead without quality claims."""
import numpy as np
from .contract import *
from .serving_analysis import table


def answers(label,repeat=None):
    result={}
    for file in (ROOT/'raw'/label).glob('*.json'):
        r=read(file)
        if isinstance(r,dict) and r.get('id') and 'output_ids' in r:
            if repeat is None or r.get('cohort_index')==repeat:result[r['id']]=r
    return result


def durations(label):
    result=[]
    for file in (ROOT/'raw'/label).glob('cohort_*.json'):
        r=read(file)
        if r.get('rank') not in [None,0]:continue
        if 'repeat' in r and r['repeat']==0 and label.startswith('replay_'):continue
        result.append(r['makespan_s'] if 'makespan_s' in r else r['end']-r['origin'])
    return result


def main():
    comparisons=[];rows=[]
    for arm in ['S4','L4','O4','T4']:
        comparisons.append((arm,'replay_'+arm+'_case1_v1','profile_'+arm+'_v1',0))
    for arm in ['SD','LD','OD','TD']:
        comparisons.append((arm,'observer_control_'+arm+'_v1','profile_live_'+arm+'_v1',None))
    for tp in [1,4]:comparisons.append((f'native_TP{tp}',f'observer_control_native_TP{tp}_v1',f'profile_native_TP{tp}_v1',None))
    for arm,control,heavy,repeat in comparisons:
        if not (ROOT/'raw'/control/'complete.json').exists() or not (ROOT/'raw'/heavy/'complete.json').exists():continue
        a=answers(control,repeat);b=answers(heavy,0 if repeat==0 else None);shared=sorted(a.keys() & b.keys())
        assert len(shared)==len(a)==len(b)
        outputs=sum(a[k]['output_ids']==b[k]['output_ids'] for k in shared)
        depth_available=all('depths' in a[k] and 'depths' in b[k] for k in shared)
        predictions_available=all('predicted_ids' in a[k] and 'predicted_ids' in b[k] for k in shared)
        depths=sum(a[k]['depths']==b[k]['depths'] for k in shared) if depth_available else None
        predictions=sum(a[k]['predicted_ids']==b[k]['predicted_ids'] for k in shared) if predictions_available else None
        x=durations(control);y=durations(heavy)
        parity=outputs==len(shared) and (depths in [None,len(shared)]) and (predictions in [None,len(shared)])
        rows.append(dict(arm=arm,control=control,heavy=heavy,N=len(shared),output_exact=outputs,depth_trace_exact=depths,
            computed_argmax_exact=predictions,control_median_service_s=float(np.median(x)),heavy_median_service_s=float(np.median(y)),
            heavy_service_overhead_percent=100*(float(np.median(y))/float(np.median(x))-1),
            status='BOUNDED_OBSERVER_PARITY' if parity else 'OBSERVER_PARITY_UNRESOLVED',
            scope='Heavy trace overhead only. Frozen-feedback output equality is not free-running quality. No observer-free task timing claim.'))
    table(ROOT/'observer_validity.csv',rows);print('OBSERVER_COMPARISONS',len(rows),flush=True)


if __name__=='__main__':main()

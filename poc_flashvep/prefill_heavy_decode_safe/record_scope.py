"""Exclude an unknown graph time only with an independent causal proof."""
import csv


def classify(records,apis,start_ns,stop_ns,fence=None,guard_ns=100000):
    excluded=[];unresolved=[]
    for row in records:
        matches=[a for a in apis if int(a['correlation'])==int(row.get('correlation') or -1)]
        reason=None
        if (row['kind']=='graph' and int(row['start_ns'])==int(row['end_ns'])==0
                and len(matches)==1):
            api=matches[0];begin,end=int(api['start_ns']),int(api['end_ns'])
            launch=(api.get('api_id') is None or
                api.get('kind')=='driver' and int(api['api_id']) in (514,515) or
                api.get('kind')=='runtime' and int(api['api_id']) in (311,312))
            if not launch:
                unresolved.append(dict(record=row,matching_APIs=matches,proof=None,
                    reason='Correlated API is not a supported host graph launch'))
                continue
            if begin>=stop_ns+guard_ns:reason='GRAPH_LAUNCH_BEGINS_AFTER_CAPTURE_END'
            elif (fence and int(fence['end']['cupti_ns'])<=start_ns
                    and end<=int(fence['begin']['cupti_ns'])-guard_ns):
                reason='GRAPH_LAUNCH_COMPLETES_BEFORE_PRE_CAPTURE_DEVICE_FENCE'
        value=dict(record=row,matching_APIs=matches,proof=reason)
        (excluded if reason else unresolved).append(value)
    return dict(passed=not unresolved,global_invalid_records=len(records),
        excluded_with_causal_proof=excluded,unresolved_records=unresolved,
        unresolved_in_or_near_capture=len(unresolved),guard_ns=guard_ns,
        scope='CPU launch timing alone never supplies GPU execution time. A prior launch also needs a later measured device fence. Unknown in-window work remains invalid. No missing record becomes idle by assumption.')


def audit(path,meta):
    with path.with_name(path.name+'.invalid.csv').open() as f:records=list(csv.DictReader(f))
    with path.open() as f:apis=[r for r in csv.DictReader(f) if r['kind'] in ('driver','runtime','internal_api')]
    assert len(records)==meta['counters']['invalid']
    return classify(records,apis,int(meta['start']['cupti_ns']),int(meta['stop']['cupti_ns']),meta.get('measurement_start_device_fence'))

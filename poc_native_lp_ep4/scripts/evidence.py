"""Paired wave statistics and trace attribution, never simulator latency."""
from collections import defaultdict
import numpy as np
from poc_native_lp_ep4.scripts.analyze import bootstrap


def estimate(values):
    values=np.asarray(values,dtype=float)
    return None if not len(values) else {'mean':float(values.mean()),'ci95':bootstrap(values),'n':len(values)}


def restart_estimate(values, restart_ids):
    """CI resampling complete engine restarts; preserve every paired repeat."""
    if len(values) != len(restart_ids):
        raise ValueError('each measurement needs its actual restart ID')
    if not values:
        return None
    groups = defaultdict(list)
    for value, restart in zip(values, restart_ids):
        groups[restart].append(float(value))
    blocks = [np.asarray(v, dtype=float) for v in groups.values()]
    sums = np.asarray([v.sum() for v in blocks])
    counts = np.asarray([len(v) for v in blocks])
    rng = np.random.default_rng(3407)
    selected = rng.integers(0, len(blocks), (5000, len(blocks)))
    samples = sums[selected].sum(1) / counts[selected].sum(1)
    return {'mean':float(np.mean(values)), 'n_repeats':len(values),
        'n_restarts':len(blocks), 'repeat_counts_by_restart':counts.tolist(),
        'ci95_restart':None if len(blocks)<2 else
            [float(v) for v in np.percentile(samples,[2.5,97.5])],
        'ci95_wave':bootstrap(values),
        'scope':'WHOLE_ENGINE_RESTART_CLUSTER_BOOTSTRAP; fixed request pool; small number of restarts'}


def restart_sensitivity(rows):
    """Headline quantities and paired policy/modality differences by restart."""
    fields=('ttft_ms','mean_itl_ms','p99_itl_ms','peak_itl_ms','peak_vsel_ms','peak_itl_ratio')
    output=[]
    by=defaultdict(list)
    for row in rows:
        if row['arm']!='DECODE_ONLY':by[(row['arm'],row['modality'])].append(row)
    for (arm,modality),items in by.items():
        output.append({'comparison':'ABSOLUTE','arm':arm,'modality':modality,
            **{key:restart_estimate([r[key] for r in items if r.get(key) is not None],
                                   [r['run'] for r in items if r.get(key) is not None])
               for key in fields}})
    pairs=defaultdict(dict)
    for row in rows:
        if row['arm']=='DECODE_ONLY':continue
        key=(row['run'],row['repeat'],row['modality'])
        if row['arm'] in pairs[key]:raise ValueError('duplicate arm/restart/repeat')
        pairs[key][row['arm']]=row
    for arm,modality in by:
        for control in ('FULL','CP512','LP_AUTO'):
            if arm==control:continue
            matched=[(run,pair) for (run,_,m),pair in pairs.items()
                     if m==modality and arm in pair and control in pair]
            if not matched:continue
            output.append({'comparison':'PAIRED_POLICY_DELTA','arm':arm,
                'control':control,'modality':modality,
                **{key:restart_estimate(
                    [p[arm][key]-p[control][key] for _,p in matched
                     if p[arm].get(key) is not None and p[control].get(key) is not None],
                    [run for run,p in matched
                     if p[arm].get(key) is not None and p[control].get(key) is not None])
                   for key in fields}})
    modality_pairs=defaultdict(dict)
    for row in rows:
        if row['arm']!='DECODE_ONLY':
            modality_pairs[(row['run'],row['repeat'],row['arm'])][row['modality']]=row
    for arm in sorted({r['arm'] for r in rows if r['arm']!='DECODE_ONLY'}):
        matched=[(run,p) for (run,_,a),p in modality_pairs.items()
                 if a==arm and set(p)=={'VISUAL_PREFILL','TOKEN_MATCHED_TEXT'}]
        output.append({'comparison':'PAIRED_VISUAL_MINUS_TEXT','arm':arm,
            **{key:restart_estimate(
                [p['VISUAL_PREFILL'][key]-p['TOKEN_MATCHED_TEXT'][key] for _,p in matched
                 if all(p[m].get(key) is not None for m in p)],
                [run for run,p in matched if all(p[m].get(key) is not None for m in p)])
               for key in fields}})
    return output


def modality_deltas(rows):
    paired=defaultdict(dict)
    for r in rows:
        if r['arm']!='DECODE_ONLY':paired[(r['arm'],r['repeat'])][r['modality']]=r
    by=defaultdict(lambda:defaultdict(list))
    for (arm,_),item in paired.items():
        if not all(k in item for k in ('VISUAL_PREFILL','TOKEN_MATCHED_TEXT')):continue
        v,t=item['VISUAL_PREFILL'],item['TOKEN_MATCHED_TEXT']
        for metric in ('ttft_ms','lm_start_to_first_token_ms','peak_vsel_ms','peak_itl_ms','p99_itl_ms','peak_itl_ratio'):
            if v.get(metric) is None or t.get(metric) is None:continue
            by[arm][metric+'_delta'].append(v[metric]-t[metric])
            if t[metric]>0:by[arm][metric+'_percent'].append(100*(v[metric]/t[metric]-1))
    return {arm:{k:estimate(v) for k,v in fields.items()} for arm,fields in by.items()}


def instrumentation_overhead(rows):
    by=defaultdict(dict)
    for r in rows:
        if r['arm']!='DECODE_ONLY':by[(r['arm'],r['modality'],r['repeat'])][r['instrumented']]=r
    result=[]
    for (arm,modality,repeat),pair in by.items():
        if set(pair)!={False,True}:continue
        clean,trace=pair[False],pair[True]
        row={'arm':arm,'modality':modality,'repeat':repeat,'scope':'INTERLEAVED_HOOK_BYPASS_VS_ACTIVE_COMPONENT_TRACE'}
        for metric in ('ttft_ms','lm_start_to_first_token_ms','mean_itl_ms','peak_itl_ms'):
            if clean.get(metric) and trace.get(metric):
                row[metric+'_overhead_percent']=100*(trace[metric]/clean[metric]-1)
        result.append(row)
    summary={}
    for key in ('ttft_ms','lm_start_to_first_token_ms','mean_itl_ms','peak_itl_ms'):
        vs=[r[key+'_overhead_percent'] for r in result if key+'_overhead_percent' in r]
        summary[key]=estimate(vs)
    strata=defaultdict(list)
    for row in result:strata[(row['arm'],row['modality'])].append(row)
    stratified=[]
    for (arm,modality),items in strata.items():
        stats={key:estimate([r[key+'_overhead_percent'] for r in items if key+'_overhead_percent' in r])
               for key in ('ttft_ms','lm_start_to_first_token_ms','mean_itl_ms','peak_itl_ms')}
        usable=all(stats[k] is not None and stats[k]['mean']<=10 for k in ('ttft_ms','mean_itl_ms'))
        stratified.append({'arm':arm,'modality':modality,'usable':usable,**stats})
    return {'pairs':result,'summary':summary,'stratified':stratified,
        'component_breakdown_usable':bool(stratified) and all(r['usable'] for r in stratified),
        'limitation':'Hook-bypass retains tiny wrapper checks. Unhooked final clean lane remains the latency headline.'}


def route_invariants(rows,prompt_by_modality):
    by=defaultdict(list)
    for r in rows:
        if r['prefill_active']:by[(r['run'],r['wave'])].append(r)
    result=[]
    for (run,wave),events in by.items():
        first=events[0];expected=prompt_by_modality[first['modality']]*8*48
        measured=sum(sum(e['injected_histogram'] or []) for e in events)
        unique=set(e['layer'] for e in events)
        result.append({'run':run,'wave':wave,'arm':first['arm'],'modality':first['modality'],
                       'logical_paths':measured,'expected_paths':expected,
                       'unique_layers':len(unique),'pass':measured==expected and len(unique)==48,
                       'scope':'LOGICAL_SOURCE_SP_PATHS; physical padding tracked separately'})
    return result


def per_layer_components(rows):
    by=defaultdict(list)
    for r in rows:
        if r['prefill_active']:by[(r['arm'],r['modality'],r['layer'])].append(r)
    result=[]
    for (arm,modality,layer),items in by.items():
        result.append({'arm':arm,'modality':modality,'layer':layer,'events':len(items),
            **{k:estimate([e[k] for e in items]) for k in ('dispatch_ms','expert_execution_ms','combine_ms','moe_ms')}})
    return result


def victim_dispatch_excess(rows,waves):
    # Exact forced decode-step matching. Unmeasured baseline steps are N/A,
    # never filled from a model or a request-specific learned constant.
    isolated={}
    for r in rows:
        if r['arm']=='DECODE_ONLY' and r['iteration'][0]=='decode_step':
            isolated[(r['wave'],r['iteration'][1],r['layer'])]=r
    result=[]
    for r in rows:
        if r['arm']=='DECODE_ONLY' or not r['prefill_active']:continue
        w=waves[r['wave']];baseline=w.get('baseline_wave')
        step=r.get('global_decode_step')
        if step is None:continue
        ref=isolated.get((baseline,step,r['layer']))
        if ref:
            result.append({'wave':r['wave'],'arm':r['arm'],'modality':r['modality'],
                'layer':r['layer'],'decode_step':step,
                'dispatch_excess_ms':r['dispatch_ms']-ref['dispatch_ms'],
                'combine_excess_ms':r['combine_ms']-ref['combine_ms'],
                'scope':'paired same forced decode position; local max-duration proxy, not synchronized global clock'})
    return result

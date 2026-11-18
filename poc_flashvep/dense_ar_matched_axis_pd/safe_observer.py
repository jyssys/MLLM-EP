"""A separate A/B/A test for profiler range placement, not a new method."""
import json
from .common import ROOT,save,write
from .campaign import job
from .run import execute
from .observer_audit import summarize


def main():
    names=[]
    runs=[]
    for mode in ('before','profile','after'):
        cfg=job('Layer',12,36,16,6.,2026111786,stage='safe_observer_'+mode,
            dataset='matched144.json',nsys_bounded_capture=mode=='profile',
            nsys_roles=['prefill','decode'],runtime_observation=mode=='profile',
            runtime_observation_roles=['prefill','decode'],clean_timing=mode!='profile',
            entry_module='poc_flashvep.dense_ar_matched_axis_pd.safe_profile_serving')
        names.append(cfg['name'])
        complete=execute(save(cfg))
        if mode!='profile':assert complete,cfg['name']
        # Preserve the original dispatcher failure if Nsight changes sessions.
        runs.append(summarize(cfg['name']))
    tests={}
    for key in runs[0]['metrics']:
        a,b,c=[r['metrics'][key] for r in runs]
        control=(a+c)/2
        tests[key]=dict(before=a,profiled=b,after=c,profiled_over_controls=b/control,
            controls_after_over_before=c/a,
            within_5pct=abs(b/control-1)<=.05 and abs(c/a-1)<=.05)
    folder=ROOT/'raw'/names[1]
    window=json.loads((folder/'window.json').read_text())
    placement={}
    for role in ('prefill','decode'):
        clock=json.loads((folder/f'nsys_clock_{role}.json').read_text())
        placement[role]=dict(start_before_measurement=clock['start']['after']<=window['start'],
            stop_after_measurement=clock['end']['before']>=window['end'],
            start_API_host_ms=clock['start_API_host_ms'],stop_API_host_ms=clock['stop_API_host_ms'])
    placement_pass=all(r['start_before_measurement'] and r['stop_after_measurement'] for r in placement.values())
    write(ROOT/'analysis/safe_observer_API_placement.json',dict(names=names,controls=runs,
        tests=tests,placement=placement,placement_regression_passed=placement_pass,
        observer_5pct_all_metrics=all(r['within_5pct'] for r in tests.values()),
        evidence='DIAGNOSTIC',clean_primary_eligible=False,
        scope='Change only observer range placement. Keep model, work, input, seed and inference adapters fixed. Original raw isolation verdicts remain unchanged. This pilot does not silently qualify Nsight child sessions or become primary performance.'))
    assert placement_pass,'Observer API calls still overlap the measured serving window'
    print('SAFE_OBSERVER_PLACEMENT_PASS',all(r['within_5pct'] for r in tests.values()),flush=True)


if __name__=='__main__':
    main()

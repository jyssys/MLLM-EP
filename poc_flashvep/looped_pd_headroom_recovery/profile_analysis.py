"""Align bounded P-only device traces with their actual serving windows."""
import sqlite3
import subprocess
from .common import *
from .analyze import records,table


def union(intervals):
    current=None
    total=0
    for a,b in sorted(intervals):
        if b<=a:continue
        if current is None:current=[a,b]
        elif a<=current[1]:current[1]=max(current[1],b)
        else:total+=current[1]-current[0];current=[a,b]
    if current is not None:total+=current[1]-current[0]
    return total


def analyze(name,clean=None):
    folder=ROOT/'raw'/name
    report=folder/'p_only_trace.nsys-rep'
    events=records(folder/'capture_events.jsonl')
    if not report.exists() or not events:
        return dict(name=name,status='PROFILE_NOT_AVAILABLE',P_kernel_busy='NOT_MEASURED')
    database=folder/'p_only_trace.sqlite'
    if not database.exists():
        result=subprocess.run(['nsys','export','--type=sqlite','--force-overwrite=false','--output='+str(database),str(report)],capture_output=True,text=True)
        write(folder/'profile_export.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:return dict(name=name,status='PROFILE_EXPORT_FAILED',P_kernel_busy='NOT_MEASURED')
    connection=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    strings={r['id']:r['value'] for r in connection.execute('SELECT * FROM StringIds')}
    markers={}
    for r in connection.execute('SELECT * FROM NVTX_EVENTS'):
        text=r['text'] or strings.get(r['textId'],'')
        if text in ('RECOVERY_CAPTURE_CLOCK','RECOVERY_CAPTURE_STOP_CLOCK'):markers[text]=dict(r)
    anchor=next((e for e in events if e['event']=='START'),None)
    if anchor is None or 'RECOVERY_CAPTURE_CLOCK' not in markers:
        return dict(name=name,status='CLOCK_ANCHOR_MISSING',P_kernel_busy='NOT_MEASURED')
    zero=markers['RECOVERY_CAPTURE_CLOCK']['start']
    origin=anchor['time']
    start=zero
    end=markers.get('RECOVERY_CAPTURE_STOP_CLOCK',{}).get('start')
    if end is None:return dict(name=name,status='CAPTURE_END_MISSING',P_kernel_busy='NOT_MEASURED')
    window=json.loads((folder/'window.json').read_text()) if (folder/'window.json').exists() else None
    if window:
        start=max(start,zero+round((window['start']-origin)*1e9))
        end=min(end,zero+round((window['end']-origin)*1e9))
    intervals=[]
    kernels=[]
    for r in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>? AND start<?',(start,end)):
        pid=r['globalPid'] if r['globalPid']==anchor['pid'] else (r['globalPid']>>24)&0xffffff
        if pid!=anchor['pid']:continue
        intervals.append((max(start,r['start']),min(end,r['end'])))
        kernels.append(dict(start_monotonic=origin+(r['start']-zero)/1e9,end_monotonic=origin+(r['end']-zero)/1e9,
            name=strings.get(r['demangledName'],str(r['demangledName'])),pid=pid))
    connection.close()
    write(folder/'aligned_kernels.json',dict(anchor=anchor,window_start=origin+(start-zero)/1e9,
        window_end=origin+(end-zero)/1e9,kernels=kernels,evidence='MEASURED_CUPTI_KERNEL_INTERVALS'))
    ratio=None
    exact=None
    if clean and window and (ROOT/'raw'/clean/'window.json').exists():
        reference=json.loads((ROOT/'raw'/clean/'window.json').read_text())
        ratio=(window['end']-window['start'])/(reference['end']-reference['start'])-1
        from ..looped_pd_atomic_prefill import evidence
        configure();evidence.ROOT=ROOT
        exact=evidence.compare(clean,name)['passed']
    result=dict(name=name,clean=clean,status='PARSED',recorded_kernel_invocations=len(kernels),
        window_start_monotonic=origin+(start-zero)/1e9,window_end_monotonic=origin+(end-zero)/1e9,
        window_seconds=(end-start)/1e9,kernel_busy_fraction=union(intervals)/(end-start) if end>start else None,
        anchor_uncertainty_s=anchor.get('anchor_upper',origin)-anchor.get('anchor_lower',origin),
        matched_wall_difference_fraction=ratio,exact_output_parity=exact,
        eligible_for_matched_clean=ratio is not None and abs(ratio)<=.05 and exact is True,
        evidence='MEASURED_BOUNDED_SELECTED_STAGE_NSYS',
        scope='Only the selected native worker is instrumented. Kernel union is not SM occupancy. Do not transfer a materially perturbed profile to clean serving.')
    write(folder/'profile_analysis.json',result)
    return result


def main():
    results=[]
    for p in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(p.read_text())
        if isinstance(cfg,dict) and cfg.get('P_only_profiler'):
            clean=cfg.get('matched_clean')
            if clean is None and 'p_only_profile' in cfg['name']:clean=cfg['name'].replace('p_only_profile','clean')
            if (ROOT/'raw'/cfg['name']).exists():results.append(analyze(cfg['name'],clean))
    table('profile_recovery_results.csv',results)
    print(json.dumps(results))


if __name__=='__main__':main()

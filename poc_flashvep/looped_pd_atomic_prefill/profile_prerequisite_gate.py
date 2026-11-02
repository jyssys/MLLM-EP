"""Avoid repeated heavy traces after a clean prerequisite already failed."""
from .common import *


def retained_profiles(entries):
    keep={v['name'] for v in entries if v['near_P_service']}
    keep.update(v['name'] for v in entries if v['state']!='PENDING')
    for chunk in {v['chunk'] for v in entries}:
        group=[v for v in entries if v['chunk']==chunk]
        if any(v['state'] in ('COMPLETE','INCOMPLETE') for v in group):continue
        # Keep one root-cause trace even if no point passes the service gate.
        root=max(group,key=lambda v:v['P_completion_rate'])
        keep.add(root['name'])
    return keep


def completion_rate(metrics, left, right):
    finished=[m['created_monotonic']+m['timings']['prefill_compute_completed'] for m in metrics
        if m['request_id'].startswith('r0_') and 'prefill_compute_completed' in m['timings']]
    return sum(left<=point<=right for point in finished)/(right-left)


def advance():
    # Only the serial queue owner can prune. It calls this between GPU jobs.
    # Unit tests, workers, and the concurrent manager cannot change the queue.
    words=Path('/proc/self/cmdline').read_bytes().split(b'\0')
    if b'poc_flashvep.looped_pd_atomic_prefill.campaign' not in words:return
    from .campaign import terminal
    state_path=ROOT/'state/extension_confirmation_20261102.json'
    if not state_path.exists():return
    state=json.loads(state_path.read_text())
    ar=state.get('qwen8',{});marker=ROOT/'state/ar_profile_stall_tail_gate_20261102.json'
    cutoff=json.loads((ROOT/'configs/extension_budget_20261102.json').read_text())['new_GPU_job_cutoff_epoch']
    if ar.get('stage')=='FINAL_RUNS' and time.time()>=cutoff-900:
        stalled=[name for name in ar['profiles'] if terminal(name)=='FAILED' and (ROOT/'raw'/name/'profile_watchdog.json').exists()]
        if stalled:
            path=ROOT/'state/skipped_jobs.json'
            skips=json.loads(path.read_text()) if path.exists() else {};omitted=[]
            for name in ar['profiles']:
                cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                if terminal(name)!='PENDING':continue
                assert not (ROOT/'raw'/name).exists()
                skips[name]=dict(time=time.time(),status='NOT_RUN_REPEATED_PROFILE_STALL_TIME_BUDGET',
                    scope='Preserve every clean start and attempted profile. A missing trace cannot satisfy the unchanged P-kernel gate. Prefer finite D validation within the final time budget.')
                omitted.append(name)
            previous=json.loads(marker.read_text()) if marker.exists() else {}
            if omitted:
                write(path,skips);write(marker,dict(time=time.time(),first_decision_time=previous.get('first_decision_time',previous.get('time',time.time())),
                    stalled=stalled,omitted=list(dict.fromkeys(previous.get('omitted',[])+omitted))))
    for model in ('ouro','huginn'):
        marker=ROOT/'state'/f'profile_prerequisite_gate_{model}_20261102.json'
        item=state.get(model,{})
        if marker.exists() or item.get('stage')!='FINAL_RUNS':continue
        center=[name for name in item['finals'] if json.loads((ROOT/'configs'/(name+'.json')).read_text()).get('rate_index')==1]
        if not center or any(terminal(name)!='COMPLETE' for name in center):continue
        entries=[]
        for name in item['profiles']:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());clean=cfg['profile_clean_reference']
            reference=json.loads((ROOT/'configs'/(clean+'.json')).read_text())
            folder=ROOT/'raw'/clean
            if not (folder/'closed_stats.json').exists() or not (folder/'window.json').exists():break
            window=json.loads((folder/'window.json').read_text());start=window['start'];span=window['last_arrival']-start
            if span<=0:break
            metrics=json.loads((folder/'closed_stats.json').read_text())['metrics']
            rate=completion_rate(metrics,start+.2*span,start+.9*span)
            refs=json.loads((ROOT/'configs'/reference.get('capacity_reference_file','capacity_references.json')).read_text())
            cell=refs[model+'_L'+str(reference['length'])];muP=max(v['mu_P'] for v in cell['chunks'].values())
            entries.append(dict(name=name,clean=clean,chunk=cfg['chunk'],state=terminal(name),
                near_P_service=rate>=.85*muP,P_completion_rate=rate,strongest_P_reference=muP))
        if len(entries)!=len(item['profiles']):continue
        keep=retained_profiles(entries);path=ROOT/'state/skipped_jobs.json'
        skips=json.loads(path.read_text()) if path.exists() else {};omitted=[]
        for value in entries:
            if value['name'] in keep or value['state']!='PENDING':continue
            assert not (ROOT/'raw'/value['name']).exists(),'Never prune an attempted namespace'
            skips[value['name']]=dict(time=time.time(),status='NOT_RUN_FAILED_STRONG_P_SERVICE_PREREQUISITE',
                source_clean=value['clean'],observed_P_completion=value['P_completion_rate'],
                required_P_completion=.85*value['strongest_P_reference'],
                scope='All three independent clean starts remain measured. Keep one root-cause trace per chunk and every service-qualified profile. No workload or acceptance threshold changes.')
            omitted.append(value['name'])
        write(path,skips);write(marker,dict(time=time.time(),entries=entries,retained=sorted(keep),omitted=omitted,
            scope='Serial queue-owner fail-fast step between jobs. Preserve completed and active traces. Do not infer missing CUDA activity or delete failed evidence.'))


if __name__=='__main__':raise SystemExit('This gate runs only inside the serial campaign between jobs.')

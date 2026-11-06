"""Keep stage-ready work, coordinator credits, and physical KV separate."""
from .common import *
from .analyze import table,pct


def main():
    summaries=[]
    for cfg_path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(cfg_path.read_text())
        if cfg.get('worker')!='rlt_worker' or cfg.get('state_probe'):continue
        folder=ROOT/'raw'/cfg['name'];window_path=folder/'window.json'
        if not window_path.exists():continue
        window=json.loads(window_path.read_text());frames=[]
        for path in folder.glob('memory_progress_*.jsonl'):
            for line in path.read_text().splitlines():
                v=json.loads(line)
                if not window['start']<=v['time']<=window['end']:continue
                frames.append(dict(time_s=v['time']-window['start'],role=v['role'],
                    queued_unretired_excluded_ready=len(v['queued_not_inflight_not_connector_blocked_ids']),
                    in_flight_submitted=len(v['all_submitted_unretired_ids']),
                    connector_blocked=len(v['connector_blocked']),
                    KV_used_blocks=v['KV_used_blocks'],KV_free_blocks=v['KV_free_blocks'],KV_total_blocks=v['KV_total_blocks'],
                    KV_active_request_blocks=v['KV_active_request_blocks'],
                    device_free_GiB=v['device_free_bytes']/2**30,allocator_GiB=v['allocator_allocated_bytes']/2**30,
                    decode_phase_requests=sum(str(q.get('phase','')) in ('RequestPhase.RECURRENT','RequestPhase.PRELUDE','RequestPhase.CODA') for q in v['requests'])))
        if frames:table(folder/'system_timeline.csv',frames)
        row=dict(name=cfg['name'],model=cfg['model'],method=cfg.get('dlocal_method'),offered_QPS=cfg['arrival_rate'],
            scope='Sampled host stage readiness excludes all unretired submissions and connector blockers. This is not measured GPU execution or guaranteed deadline slack.')
        for role in ('prefill','decode'):
            selected=[v for v in frames if v['role']==role]
            if not selected:continue
            label='P' if role=='prefill' else 'D'
            for field in ('queued_unretired_excluded_ready','KV_free_blocks','device_free_GiB'):
                row[label+'_'+field+'_p50']=pct([v[field] for v in selected],50)
                row[label+'_'+field+'_p95']=pct([v[field] for v in selected],95)
                row[label+'_'+field+'_min']=min(v[field] for v in selected)
        credits=[]
        path=folder/'queue_observations.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                v=json.loads(line)
                if not window['start']<=v['time']<=window['end']:continue
                for peer in v['worker_credits'].values():
                    if peer['role']=='decode':credits.append(dict(time_s=v['time']-window['start'],
                        reserved_blocks=peer['reserved_blocks'],total_blocks=peer['total_blocks'],reserved_slots=peer['slots']))
        if credits:
            table(folder/'D_coordinator_credits.csv',credits)
            row['D_coordinator_free_blocks_min']=min(v['total_blocks']-v['reserved_blocks'] for v in credits)
            row['D_reserved_slots_max']=max(v['reserved_slots'] for v in credits)
        summaries.append(row)
    table(ROOT/'system_summary.csv',summaries)
    print(json.dumps(dict(runs=len(summaries))))


if __name__=='__main__':main()

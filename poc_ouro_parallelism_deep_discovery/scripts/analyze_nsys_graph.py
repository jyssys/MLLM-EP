"""Export only this campaign's traced GPU work. Keep observer limits explicit."""
import argparse
import collections
import csv
import html
import sqlite3
from poc_ouro_parallelism_deep_discovery.contract import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();run=args.run;target=run/'sanitized_timeline';assert not target.exists()
    c=sqlite3.connect(run/'native_graph.sqlite');c.row_factory=sqlite3.Row
    observed=read(run/'profile_observation.json');owners=read(run/'ownership.json')
    own={r['pid']:r for r in owners}
    processes={r['globalPid']:r['pid'] for r in c.execute('SELECT globalPid,pid FROM PROCESSES') if r['pid'] in own}
    assert len(processes)==len(owners)
    strings={r['id']:r['value'] for r in c.execute('SELECT id,value FROM StringIds')}
    ranges=[r for r in c.execute('SELECT start,end,text FROM NVTX_EVENTS') if r['text']=='OBSERVED_REQUEST']
    assert len(ranges)==1 and ranges[0]['end'] is not None
    lo,hi=ranges[0]['start'],ranges[0]['end'];records=[]
    for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','KERNEL'),('CUPTI_ACTIVITY_KIND_GRAPH_TRACE','GRAPH_ENVELOPE'),('CUPTI_ACTIVITY_KIND_MEMCPY','MEMCPY')]:
        for row in c.execute('SELECT * FROM '+table):
            if row['globalPid'] not in processes:continue
            if row['end']<=lo or row['start']>=hi:continue
            pid=processes[row['globalPid']];owner=own[pid]
            physical=int(owner['CUDA_VISIBLE_DEVICES'].split(',')[owner['rank']])
            name=strings[row['demangledName']] if kind=='KERNEL' else kind
            records.append(dict(physical_gpu=physical,logical_rank=owner['rank'],UUID=owner['GPU_uuid'],
                kind=kind,name=name,stream=row['streamId'],
                start_ms=(max(row['start'],lo)-lo)/1e6,end_ms=(min(row['end'],hi)-lo)/1e6,
                duration_ms=(min(row['end'],hi)-max(row['start'],lo))/1e6,
                bytes=row['bytes'] if kind=='MEMCPY' else None,
                interpretation='Trace interval, not non-overlapped E2E attribution. A graph envelope can include internal waits.'))
    assert {r['physical_gpu'] for r in records}==set(observed['physical_gpus'])
    target.mkdir()
    with (target/'gpu_intervals.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    summaries=[]
    for owner in owners:
        rows=[r for r in records if r['logical_rank']==owner['rank']]
        points=sorted({x for r in rows for x in [r['start_ms'],r['end_ms']]})
        union=0.
        for a,b in zip(points,points[1:]):
            if any(r['start_ms']<b and r['end_ms']>a for r in rows):union+=b-a
        summaries.append(dict(rank=owner['rank'],physical_gpu=observed['physical_gpus'][owner['rank']],
            kernel_records=sum(r['kind']=='KERNEL' for r in rows),
            graph_records=sum(r['kind']=='GRAPH_ENVELOPE' for r in rows),
            graph_sum_ms=sum(r['duration_ms'] for r in rows if r['kind']=='GRAPH_ENVELOPE'),
            traced_envelope_union_ms=union,request_NVTX_ms=(hi-lo)/1e6,
            uncovered_interval_ms=(hi-lo)/1e6-union,
            meaning='Uncovered intervals can include host work, waiting, untraced work, or profiler overhead. They are not proved removable idle.'))
    width=1200;height=80+len(owners)*65;span=(hi-lo)/1e6;left=155;scale=(width-left-30)/span
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="10" y="20" font-size="14">Actual Nsight diagnostic: blue graph envelopes, orange kernels, green copies. Not headline timing.</text>']
    colors={'GRAPH_ENVELOPE':'#2a6fba','KERNEL':'#e4892d','MEMCPY':'#269b56'}
    for i,owner in enumerate(owners):
        y=45+i*65;physical=observed['physical_gpus'][owner['rank']]
        svg.append(f'<text x="10" y="{y+17}" font-size="13">GPU{physical} / rank{owner["rank"]}</text>')
        for r in records:
            if r['logical_rank']!=owner['rank']:continue
            x=left+r['start_ms']*scale;w=max(.2,(r['end_ms']-r['start_ms'])*scale)
            svg.append(f'<rect x="{x:.3f}" y="{y}" width="{w:.3f}" height="24" fill="{colors[r["kind"]]}"><title>{html.escape(r["name"])} {r["duration_ms"]:.6f} ms</title></rect>')
        svg.append(f'<text x="{left}" y="{y+43}" font-size="11">0 to {span:.3f} ms, aligned to parent OBSERVED_REQUEST NVTX range</text>')
    svg.append('</svg>');(target/'timeline.svg').write_text('\n'.join(svg)+'\n')
    write(target/'summary.json',dict(records=len(records),ranks=summaries,
        observer_vs_before=observed['observer_vs_before_fraction'],observer_vs_after=observed['observer_vs_after_fraction'],
        validity='UNVALIDATED_FOR_HEADLINE_COMPONENT_PERCENTAGES' if max(abs(observed['observer_vs_before_fraction']),abs(observed['observer_vs_after_fraction']))>.1 else 'DIAGNOSTIC_ONLY',
        mapping='Own worker PID links to recorded UUID and CUDA_VISIBLE_DEVICES. Nsight device IDs are logical, not TARGET_INFO_GPU physical IDs.',
        privacy='Only owned GPU intervals and one request range are exported. Full trace retains machine process inventories locally and is excluded from Git.',
        raw_sqlite_sha256=sha(run/'native_graph.sqlite'),raw_report_sha256=sha(run/'native_graph.nsys-rep'),
        critical_path='PARTIAL. Graph internals remain opaque. No attribution to a new recurrent-specific removable cost.'))
    print('SANITIZED_TIMELINE',len(records),'records',len(owners),'ranks')

if __name__=='__main__':main()

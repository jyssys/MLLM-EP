"""Write queue, stage, and transport brackets without claiming exclusive GPU work."""
import collections
import csv
import json
import numpy as np
from .analyze import read
from .contract import ROOT,write


def main():
    stages=[];edges=[];clocks=[];paths=[]
    for f in sorted((ROOT/'raw').glob('*v5*/*.json.gz')):
        d=read(f)
        if not d['trace']:continue
        for c in d['clock_checks']:clocks.append(dict(file=str(f.relative_to(ROOT)),**c))
        for out in d['outputs']:
            es=[e for e in d['events'] if e['id']==out['id'] and e['step']==0]
            paths.append(dict(file=str(f.relative_to(ROOT)),arm=d['placement'],R=d['depth'],C=d['C'],
                id=out['id'],prompt_tokens=len(out['input_ids']),
                prefill_path_host_ms=(max(e['host_end_s'] for e in es)-min(e['host_start_s'] for e in es))*1000,
                initial_entry_queue_ms=min(es,key=lambda e:e['host_start_s'])['queue_ms'],
                TTFT_host_ms=out['token_times_s'][0]*1000,
                evidence='LIVE_MEASURED trace lane; path excludes initial admission wait, includes internal queues'))
        grouped=collections.defaultdict(list);transport=collections.defaultdict(list)
        for e in d['events']:
            grouped[(e['rank'],'prefill' if e['step']==0 else 'decode',e['loop'])].append(e)
            if e['transfer_kind']:
                transport[(e['source_rank'],e['rank'],e['transfer_kind'])].append(e)
        for (rank,phase,loop),es in grouped.items():
            stages.append(dict(file=str(f.relative_to(ROOT)),arm=d['placement'],R=d['depth'],C=d['C'],
                rank=rank,physical_GPU=rank+4,phase=phase,loop=loop+1,stage_count=len(es),
                q_tokens_min=min(e['q_tokens'] for e in es),q_tokens_max=max(e['q_tokens'] for e in es),
                compute_bracket_total_ms=sum(e['compute_ms'] for e in es),
                compute_bracket_P50_ms=float(np.median([e['compute_ms'] for e in es])),
                head_bracket_total_ms=sum(e['head_ms'] for e in es),
                copy_bracket_total_ms=sum(e['transfer_ms'] for e in es),
                ready_queue_total_ms=sum(e['queue_ms'] for e in es),
                ready_queue_P90_ms=float(np.quantile([e['queue_ms'] for e in es],.9)),
                evidence='GPU_BRACKET; ready queue is host time'))
        for (src,dst,kind),es in transport.items():
            edges.append(dict(file=str(f.relative_to(ROOT)),arm=d['placement'],R=d['depth'],C=d['C'],
                source=src,destination=dst,kind=kind,count=len(es),bytes=sum(e['transfer_bytes'] for e in es),
                copy_bracket_total_ms=sum(e['transfer_ms'] for e in es),
                evidence='Executed payload accounting + GPU_BRACKET. Not pure link bandwidth.'))
    for name,rows in [('stage_metrics',stages),('transport_edges',edges),('clock_validation',clocks),('prefill_paths',paths)]:
        write(ROOT/f'{name}.json',rows)
        if rows:
            with (ROOT/f'{name}.csv').open('w',newline='') as out:
                writer=csv.DictWriter(out,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print('TRACE_TABLES',len(stages),len(edges),len(clocks))


if __name__=='__main__':main()

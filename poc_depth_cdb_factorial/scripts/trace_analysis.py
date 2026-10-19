"""Separate measured stream intervals from whole-GPU idle and model estimates."""
import collections
import numpy as np
from poc_depth_cdb_factorial.contract import *
from poc_depth_cdb_factorial.scripts.analyze import records,parity,write_csv

def main():
    cases=list(records())
    comparisons=[]
    rows=[]
    for output,source,d,ranks,values in cases:
        if not d.get('trace') or d['case']['kind']!='speed':continue
        case=d['case']
        peer=next((r for r in cases if r[2]['case']['kind']=='speed' and not r[2].get('trace')
            and r[2]['case']['C']==case['C'] and r[2]['case']['arm']==case['arm']
            and r[2]['case']['repeat']==0),None)
        if peer:
            clean_finish=max(r['finish_s'] for r in peer[4])-peer[2]['first_submitted_s']
            trace_finish=max(r['finish_s'] for r in values)-d['first_submitted_s']
            equal=parity(peer[4],values)
            valid=equal['token_exact']==equal['N'] and equal['depth_exact']==equal['N']
            comparisons.append(dict(C=case['C'],arm=case['arm'],observed_trace_clean_delta_pct=100*(trace_finish/clean_finish-1),
                parity_matched_trace_delta_pct=100*(trace_finish/clean_finish-1) if valid else None,
                observer_validity='OUTPUT_AND_DEPTH_PARITY' if valid else 'UNVALIDATED_OUTPUT_DRIFT',
                causal_observer_overhead='PARTIAL. One paired run cannot separate observer cost from run-to-run variation.',
                clean_s=clean_finish,trace_s=trace_finish,**equal,
                repetitions=1,scope='One paired trace/clean run. Not independent request-level timing trials.'))
        for rank,raw in ranks:
            bystage=collections.defaultdict(float)
            for stage in raw['stages']:bystage[stage['stage']]+=stage['gpu_ms']
            events=sorted(raw['stages'],key=lambda r:r.get('gpu_start_s',r['host_start_s']))
            gaps=[]
            stream_gap_with_active=0.
            samples=raw['queue_timeline']
            for left,right in zip(events,events[1:]):
                if 'gpu_end_s' not in left:continue
                start,finish=left['gpu_end_s'],right['gpu_start_s']
                if finish<=start:continue
                previous=[s for s in samples if s['t']<=(start+finish)/2]
                status=previous[-1] if previous else {}
                if status.get('active',0)+status.get('waiting',0)>0:
                    stream_gap_with_active+=(finish-start)*1000
                gaps.append((finish-start)*1000)
            depths=[e['depth'] for r in raw['outputs'] for e in r['depths']]
            predictions=read(ROOT/'artifacts/speed_reference_depth_work.json')
            actual=sum(depths)
            predicted=sum(predictions[r['id']] for r in raw['outputs'])
            core_stages=[s for s in raw['stages'] if s['stage']=='compute_recurrent_batch']
            valid_rows=sum(s['batch'] for s in core_stages)
            padded_rows=sum(s.get('occupancy',{}).get('core_padded_rows',s['batch']) for s in core_stages)
            rows.append(dict(case=case['name'],C=case['C'],arm=case['arm'],rank=rank,
                physical_GPU=raw['outputs'][0]['gpu'] if raw['outputs'] else 4+rank,
                requests=len(raw['outputs']),actual_loop_work=actual,predicted_loop_work=predicted,
                work_prediction_error_pct=100*(actual/predicted-1) if predicted else None,
                generated_tokens=sum(r['generated_tokens'] for r in raw['outputs']),
                rank_completion_s=max([r['finish_s'] for r in raw['outputs']] or [0])-d['first_submitted_s'],
                prefill_GPU_ms=bystage['stage_prefill_batch'],core_GPU_ms=bystage['compute_recurrent_batch'],
                prelude_GPU_ms=bystage['compute_prelude_batch'],coda_GPU_ms=bystage['stage_coda_batch'],
                exit_KV_copy_GPU_ms=bystage['exit_KV_copy'],
                GPU_stream_stage_ms=sum(bystage.values()),
                active_core_rows=valid_rows,padded_core_rows=padded_rows,
                graph_padding_row_amplification=padded_rows/valid_rows if valid_rows else None,
                measured_stream_gap_with_sampled_active_ms=stream_gap_with_active,
                whole_GPU_idle_with_backlog='UNRESOLVED. Other CUDA streams and eligibility waits can occupy a measured compute-stream gap.',
                max_stream_gap_ms=max(gaps or [0]),
                timestamp_anchor_uncertainty_ms=raw.get('GPU_anchor_uncertainty_ms'),
                max_sampled_active=max([s['active'] for s in samples] or [0]),
                max_sampled_waiting=max([s['waiting'] for s in samples] or [0]),
                peak_sampled_KV_blocks=max([s.get('used_KV_blocks',0) for s in samples] or [0]),
                refill_events=raw['stats']['scheduler_refills'],core_batch_histogram={k:v for k,v in raw['batch_histogram'].items() if k.startswith('compute_recurrent_batch')},
                sampled_reservation_blocked=sum(s.get('reservation_blocks_admission',False) for s in samples),
                blocked_on_KV='Reserve-admission state is sampled. Exact continuous blocked duration is UNRESOLVED.'))
    write(ROOT/'artifacts/observer_parity.json',comparisons)
    write(ROOT/'artifacts/rank_time_analysis.json',rows)
    write_csv(ROOT/'artifacts/rank_time_analysis.csv',rows)
    write_csv(ROOT/'artifacts/observer_parity.csv',comparisons)
    print('TRACE_ANALYSIS',len(rows),'rank traces')

if __name__=='__main__':main()

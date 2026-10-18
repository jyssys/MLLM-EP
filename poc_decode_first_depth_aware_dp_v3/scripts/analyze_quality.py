"""Versioned quality continuity. Do not apply the superseded v2 quality stop."""
import statistics
import numpy as np
from poc_async_recurrent_pp.scoring import score
from poc_decode_first_depth_aware_dp_v3.contract import *
from poc_decode_first_depth_aware_dp_v3.scripts.analyze_discovery import interval

def main():
    wanted={r['id']:r for r in read(OLD_ROOT/'artifacts/quality_protocol_v3_pilot32.json')['rows'] if r['split']=='pilot'}
    old_predictions={(r['id'],r['policy']):r for r in read(OLD_ROOT/'artifacts/quality_pilot_analysis_v2/quality_predictions.json')}
    raw_old={}
    for folder in OLD_ROOT.glob('outputs/quality_dynamic_*'):
        if 'cap8192' in folder.name:continue
        for p in folder.glob('*.json'):
            r=read(p)
            if isinstance(r,dict) and 'depth_events' in r:raw_old[r['id']]=r
    rows=[]
    for part in range(4):
        folder=ROOT/f'outputs/quality8192_part{part}_v1'
        assert read(folder/'status.json')['success']
        for p in folder.glob('*.json'):
            r=read(p)
            if not isinstance(r,dict) or 'depth_events' not in r:continue
            row=wanted[r['id']]
            stopped=r['finish_reason']=='eos' or (row['dataset']=='gsm8k' and r['output_length']<8192)
            result=score(row,dict(text=r['text'],finish_reason='eos' if stopped else 'length'))
            old=raw_old[r['id']]
            n=min(len(old['output_ids']),len(r['output_ids']))
            events=r['depth_events']
            rows.append(dict(id=r['id'],dataset=r['dataset'],new_correct=result['answer_matches'],
                new_parsed=result['parsed'],new_complete=stopped,new_output_tokens=r['output_length'],
                old_dynamic_correct=old_predictions[r['id'],'dynamic']['answer_matches'],
                old_fixed_correct=old_predictions[r['id'],'fixed']['answer_matches'],
                token_prefix_exact=r['output_ids'][:n]==old['output_ids'][:n],
                compared_prefix_tokens=n,depths=[e['depth'] for e in events],
                mean_depth=r['mean_depth'],output_path=str(p)))
    assert {r['id'] for r in rows}==set(wanted)
    result=[]
    for domain in ['gsm8k','math500']:
        values=sorted((r for r in rows if r['dataset']==domain),key=lambda r:r['id'])
        delta=[100*(int(r['new_correct'])-int(r['old_dynamic_correct'])) for r in values]
        drop=[100*(int(r['old_fixed_correct'])-int(r['new_correct'])) for r in values]
        work=sum(sum(r['depths']) for r in values)
        full=4*sum(len(r['depths']) for r in values)
        result.append(dict(dataset=domain,n=len(values),new_correct=sum(r['new_correct'] for r in values),
            old_dynamic_correct=sum(r['old_dynamic_correct'] for r in values),
            old_fixed_correct=sum(r['old_fixed_correct'] for r in values),
            new_minus_old_dynamic_pp=statistics.mean(delta),CI95_new_minus_old_dynamic_pp=interval(delta),
            old_fixed_minus_new_dynamic_pp=statistics.mean(drop),CI95_fixed_minus_new_pp=interval(drop),
            parse_failures=sum(not r['new_parsed'] for r in values),
            cap_truncations=sum(not r['new_complete'] for r in values),
            exact_prefix_requests=sum(r['token_prefix_exact'] for r in values),
            mean_request_depth=statistics.mean(r['mean_depth'] for r in values),
            actual_loop_work_saved_percent=100*(1-work/full)))
    observer={}
    root=ROOT/'outputs'
    clean={r['id']:r for p in (root/'observer_graph_clean_v1').glob('*.json')
        if isinstance((r:=read(p)),dict) and 'depth_events' in r}
    for mode in ['graph_trace','eager_clean']:
        vals={r['id']:r for p in (root/f'observer_{mode}_v1').glob('*.json')
            if isinstance((r:=read(p)),dict) and 'depth_events' in r}
        assert set(vals)==set(clean)
        deltas=[100*(vals[i]['decode_ms']/clean[i]['decode_ms']-1) for i in clean]
        observer[mode]=dict(n=len(vals),token_exact=sum(vals[i]['output_ids']==clean[i]['output_ids'] for i in clean),
            depth_sequence_exact=sum([e['depth'] for e in vals[i]['depth_events']]==[e['depth'] for e in clean[i]['depth_events']] for i in clean),
            mean_decode_delta_percent=statistics.mean(deltas),median_decode_delta_percent=statistics.median(deltas),
            CI95_mean_decode_delta_percent=interval(deltas))
    write(ROOT/'artifacts/quality_recheck_summary_v2.json',dict(rows=result,selection='INCUMBENT_CONTINUE',
        warning='QUALITY_WARNING',challenger='No independent runnable better-quality candidate. The official CDB source is the incumbent.',
        old_primary_cap=4096,new_cap=8192,bootstrap=20000,seed=20261018,sample_unit='Paired question ID sorted lexicographically',
        analysis_version_note='v1 retained. v2 fixes bootstrap row order to the original sorted question-ID convention. Counts and decisions do not change.',
        paper_match='PARTIAL_PROTOCOL_MATCH',
        comparison='Recheck of the same implementation. Old EOS outputs and capped prefixes retain their original status.',
        observer=observer))
    write(ROOT/'artifacts/quality_recheck_predictions.json',rows)
    print(result,observer)

if __name__=='__main__':main()

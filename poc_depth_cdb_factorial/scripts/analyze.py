"""Analyze measured cohorts. Keep quality, trace, and modeled evidence separate."""
import argparse
import collections
import csv
import gzip
import math
import numpy as np
from poc_depth_cdb_factorial.contract import *

def quantile(values,q):
    return float(np.quantile(values,q)) if len(values) else None

def cv(values):
    return float(np.std(values)/np.mean(values)) if np.mean(values)>0 else 0.

def paired_ratio(a,b,seed=20261019):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    assert a.shape==b.shape and a.size>0
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(a),size=(2000,len(a)))
    draws=np.median(a[idx],axis=1)/np.median(b[idx],axis=1)
    return dict(speedup=float(np.median(a)/np.median(b)),CI95=np.quantile(draws,[.025,.975]).tolist(),
        n_paired_repeats=len(a),statistic='Ratio of median cohort makespans',bootstrap_unit='Paired whole-cohort repeats')

def paired_accuracy(a,b):
    delta=(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))*100
    rng=np.random.default_rng(20261019)
    idx=rng.integers(0,len(delta),size=(2000,len(delta)))
    return dict(delta_pp=float(delta.mean()),CI95_pp=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist(),
        N=len(delta),bootstrap_unit='Paired questions')

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)

def records():
    for output in sorted((ROOT/'outputs').iterdir()):
        if not (output/'replicas.json').exists():continue
        sources=sorted({Path(str(p).removesuffix('.gz')) for p in output.glob('*.json*')})
        for source in sources:
            if not source.name.startswith(('smoke_','speed_','trace_speed_','quality_','sensitivity_','order_control_')):continue
            d=read(source)
            if 'case' not in d or not isinstance(d['case'],dict):continue
            case=d['case']
            ranks=[]
            for rankdir in sorted(output.glob('rank_*')):
                raw=rankdir/(case['name']+'.json')
                if not raw.exists() and not Path(str(raw)+'.gz').exists():continue
                ranks.append((int(rankdir.name[5:]),read(raw)))
            values=[r for _,rank in ranks for r in rank['outputs']]
            assert set(r['id'] for r in values)==set(case['ids']),(source,len(values))
            yield output,source,d,ranks,values

def summarize(source,d,ranks,values):
    case=d['case']
    first=d['first_submitted_s']
    finish=max(r['finish_s'] for r in values)
    duration=finish-first
    e2e=[(r['finish_s']-first)*1000 for r in values]
    ttft=[(r['first_token_s']-first)*1000 for r in values]
    decode=[(r['finish_s']-r['first_token_s'])*1000 for r in values]
    tpots=[t/max(1,r['generated_tokens']-1) for t,r in zip(decode,values)]
    itls=[(b-a)*1000 for r in values for a,b in zip(r['token_host_times_s'],r['token_host_times_s'][1:])]
    loops=[sum(e['depth'] for r in rank['outputs'] for e in r['depths']) for _,rank in ranks]
    totals=[sum(r['generated_tokens'] for r in rank['outputs']) for _,rank in ranks]
    finishes=[max([r['finish_s']-first for r in rank['outputs']] or [0]) for _,rank in ranks]
    hist=collections.Counter()
    for _,rank in ranks:hist.update(rank['batch_histogram'])
    batches={int(k.rsplit(':',1)[1]):v for k,v in hist.items() if k.startswith('compute_recurrent_batch:')}
    core_mean=sum(k*v for k,v in batches.items())/max(1,sum(batches.values()))
    queued=[(r['first_scheduled_s']-first)*1000 for r in values]
    gpu_times=[sum(s['gpu_ms'] for s in rank['stages'] if s['stage']=='compute_recurrent_batch') for _,rank in ranks]
    row=dict(case=case['name'],kind=case['kind'],arm=case['arm'],C=case['C'],repeat=case['repeat'],
        trace=d.get('trace',False),validation=d.get('validation',False),N=len(values),makespan_s=duration,
        request_per_s=len(values)/duration,output_tokens=sum(totals),tokens_per_s=sum(totals)/duration,
        e2e_P50_ms=quantile(e2e,.5),e2e_P95_ms=quantile(e2e,.95),TTFT_P50_ms=quantile(ttft,.5),TTFT_P95_ms=quantile(ttft,.95),
        TPOT_mean_ms=float(np.mean(tpots)),ITL_P50_ms=quantile(itls,.5),ITL_P95_ms=quantile(itls,.95),
        decode_P95_ms=quantile(decode,.95),queue_P50_ms=quantile(queued,.5),queue_P95_ms=quantile(queued,.95),
        goodput_req_per_s=sum((r['finish_s']-first)<=max(10.,r['generated_tokens']*.2) for r in values)/duration,
        actual_recurrent_steps=sum(loops),rank_depth_work=loops,rank_generated_tokens=totals,
        depth_work_CV=cv(loops),rank_finish_s=finishes,rank_finish_CV=cv(finishes),
        depth_work_max_min=max(loops)/min(loops) if min(loops)>0 else None,
        depth_work_max_mean=max(loops)/float(np.mean(loops)) if np.mean(loops)>0 else None,
        rank_finish_max_min=max(finishes)/min(finishes) if min(finishes)>0 else None,
        rank_finish_max_mean=max(finishes)/float(np.mean(finishes)) if np.mean(finishes)>0 else None,
        rank_without_local_request_tail_s=[duration-t for t in finishes],
        rank_core_GPU_ms=gpu_times if d.get('trace') else 'TRACE_LANE_ONLY',
        core_GPU_CV=cv(gpu_times) if d.get('trace') else None,
        core_batch_mean=core_mean,core_batch_histogram=batches,mean_empty_core_slots=16-core_mean,
        refill_events=sum(rank['stats']['scheduler_refills'] for _,rank in ranks),
        prefill_batches=sum(rank['stats']['prefill_batches'] for _,rank in ranks),
        recurrent_graph_captures=sum(rank['stats']['recurrent_graph_captures'] for _,rank in ranks),
        peak_HBM_GiB=max(rank['peak_memory'] for _,rank in ranks)/2**30,
        peak_reserved_GiB=max(rank['peak_reserved'] for _,rank in ranks)/2**30,
        KV_GiB=max(rank['cache_bytes'] for _,rank in ranks)/2**30,
        finished_EOS=sum(r['finish_reason']=='eos' for r in values),
        finished_stop=sum(r['finish_reason']=='stop' for r in values),
        truncated=sum(r['finish_reason']=='length' for r in values),source=str(source.relative_to(ROOT)))
    return row

def parity(a,b):
    left={r['id']:r for r in a}
    right={r['id']:r for r in b}
    ids=sorted(left.keys()&right.keys())
    tokens=sum(left[i]['output_ids']==right[i]['output_ids'] for i in ids)
    depths=sum([e['depth'] for e in left[i]['depths']]==[e['depth'] for e in right[i]['depths']] for i in ids)
    total=agree=0
    common_history_positions=common_history_disagreements=0
    examples=[]
    for rid in ids:
        x,y=left[rid]['output_ids'],right[rid]['output_ids']
        total+=max(len(x),len(y))
        agree+=sum(a==b for a,b in zip(x,y))
        first=next((j for j,(a,b) in enumerate(zip(x,y)) if a!=b),min(len(x),len(y)))
        common_history_positions+=first
        if first<min(len(x),len(y)):
            common_history_positions+=1
            common_history_disagreements+=1
        if x!=y and len(examples)<3:
            dx=[e['depth'] for e in left[rid]['depths']]
            dy=[e['depth'] for e in right[rid]['depths']]
            first_depth=next((j+1 for j,(a,b) in enumerate(zip(dx,dy)) if a!=b),None)
            examples.append(dict(id=rid,first_divergent_token=first,left_length=len(x),right_length=len(y),
                left_token=x[first] if first<len(x) else None,right_token=y[first] if first<len(y) else None,
                first_divergent_decode_depth_output_index=first_depth))
    return dict(N=len(ids),token_exact=tokens,token_exact_pct=100*tokens/max(1,len(ids)),
        depth_exact=depths,depth_exact_pct=100*depths/max(1,len(ids)),
        free_running_positional_token_agreement_pct=100*agree/max(1,total),
        common_history_greedy_top1_positions=common_history_positions,
        common_history_greedy_top1_disagreements=common_history_disagreements,
        common_history_greedy_top1_disagreement_pct=100*common_history_disagreements/max(1,common_history_positions),
        teacher_aligned_top1='NOT_MEASURED. Free-running position agreement is not teacher-aligned logit agreement.',
        first_divergences=examples)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quality',action='store_true')
    args=parser.parse_args()
    cases=list(records())
    metrics=[summarize(source,d,ranks,values) for _,source,d,ranks,values in cases]
    write(ROOT/'artifacts/metrics_raw.json',metrics)
    write_csv(ROOT/'artifacts/metrics_raw.csv',metrics)
    clean=[r for r in metrics if r['kind']=='speed' and not r['trace'] and not r['validation']]
    points=[]
    incremental=[]
    for C in [4,8,16,32,64]:
        table={a:sorted([r for r in clean if r['C']==C and r['arm']==a],key=lambda r:r['repeat']) for a in ARMS}
        if not all(table.values()):continue
        for a in ARMS:
            reps=sorted(set(r['repeat'] for r in table['A0'])&set(r['repeat'] for r in table[a]))
            baseline=[next(r['makespan_s'] for r in table['A0'] if r['repeat']==i) for i in reps]
            actual=[next(r['makespan_s'] for r in table[a] if r['repeat']==i) for i in reps]
            if not reps:continue
            point=dict(C=C,arm=a,median_makespan_s=float(np.median(actual)),**paired_ratio(baseline,actual),
                median_peak_HBM_GiB=float(np.median([r['peak_HBM_GiB'] for r in table[a]])),
                median_TTFT_ms=float(np.median([r['TTFT_P50_ms'] for r in table[a]])),
                median_TPOT_ms=float(np.median([r['TPOT_mean_ms'] for r in table[a]])),
                median_P95_E2E_ms=float(np.median([r['e2e_P95_ms'] for r in table[a]])),
                median_core_batch=float(np.median([r['core_batch_mean'] for r in table[a]])),
                median_refill_events=float(np.median([r['refill_events'] for r in table[a]])),
                median_depth_CV=float(np.median([r['depth_work_CV'] for r in table[a]])),
                median_finish_CV=float(np.median([r['rank_finish_CV'] for r in table[a]])))
            points.append(point)
        reps=set.intersection(*(set(r['repeat'] for r in table[a]) for a in ['A1','A2','A3','A4']))
        if reps:
            times={a:[next(r['makespan_s'] for r in table[a] if r['repeat']==i) for i in sorted(reps)] for a in ['A1','A2','A3','A4']}
            row=dict(C=C,no_refill_hindsight=paired_ratio(times['A1'],times['A2']),
                refill_hindsight=paired_ratio(times['A3'],times['A4']),
                CDB_without_hindsight=paired_ratio(times['A1'],times['A3']),
                CDB_with_hindsight=paired_ratio(times['A2'],times['A4']))
            logs=np.log(np.asarray(times['A1'])/times['A2'])-np.log(np.asarray(times['A3'])/times['A4'])
            rng=np.random.default_rng(20261019)
            idx=rng.integers(0,len(logs),size=(2000,len(logs)))
            row['interaction_log']=dict(mean=float(logs.mean()),CI95=np.quantile(logs[idx].mean(axis=1),[.025,.975]).tolist(),
                direction='Positive means greater hindsight benefit without refill. Negative means greater benefit with refill.')
            incremental.append(row)
    write(ROOT/'artifacts/scaling_summary.json',points)
    write_csv(ROOT/'artifacts/scaling_summary.csv',points)
    write(ROOT/'artifacts/factorial_incremental.json',incremental)
    parity_rows=[]
    observations={(d['case']['kind'],d['case']['C'],d['case']['repeat'],d['case']['arm']):values
        for _,_,d,_,values in cases if not d.get('trace') or d['case']['kind']=='smoke'}
    for kind in ['smoke','speed']:
        for C in [4,8,16,32,64]:
            for repeat in range(5):
                for a,b in [('A0','A1'),('A0','A2'),('A0','A3'),('A0','A4'),('A1','A2'),('A3','A4')]:
                    x,y=observations.get((kind,C,repeat,a)),observations.get((kind,C,repeat,b))
                    if x is not None and y is not None:parity_rows.append(dict(kind=kind,C=C,repeat=repeat,left=a,right=b,**parity(x,y)))
    write(ROOT/'artifacts/output_parity.json',parity_rows)
    repeat_parity=[]
    for C in [4,8,16,32,64]:
        for arm in ARMS:
            base=observations.get(('speed',C,0,arm))
            for repeat in [1,2,3,4]:
                other=observations.get(('speed',C,repeat,arm))
                if base is not None and other is not None:
                    repeat_parity.append(dict(C=C,arm=arm,left_repeat=0,right_repeat=repeat,**parity(base,other)))
    write(ROOT/'artifacts/within_arm_repeat_parity.json',repeat_parity)
    timelines=[]
    placements=[]
    for output,source,d,ranks,values in cases:
        for item in d['assignments']:placements.append(dict(case=d['case']['name'],arm=d['case']['arm'],**item))
        if not d.get('trace'):continue
        for rank,raw in ranks:
            timelines.extend(dict(case=d['case']['name'],arm=d['case']['arm'],rank=rank,C=d['case']['C'],**s) for s in raw['stages'])
    for name,rows in [('placements.jsonl',placements),('rank_timeline.jsonl',timelines)]:
        with (ROOT/'artifacts'/name).open('w') as stream:
            for r in rows:stream.write(json.dumps(r)+'\n')
    write_csv(ROOT/'artifacts/rank_timeline_summary.csv',[dict(case=r['case'],rank=r['rank'],stage=r['stage'],gpu_ms=r['gpu_ms'],
        host_start_s=r['host_start_s'],batch=r['batch'],gpu_start_s=r.get('gpu_start_s'),gpu_end_s=r.get('gpu_end_s')) for r in timelines])
    if args.quality:
        from poc_async_recurrent_pp.scoring import score
        frozen=read(ROOT/'artifacts/workload_frozen.json')
        questions={r['id']:r for r in frozen['quality']}
        quality={a:{} for a in ARMS}
        for _,_,d,_,values in cases:
            if d['case']['kind']=='quality':quality[d['case']['arm']].update({r['id']:r for r in values})
        scored=[]
        cached_path=ROOT/'artifacts/quality_scores_cache.json'
        cached=read(cached_path) if cached_path.exists() else {}
        for arm,values in quality.items():
            for rid,r in values.items():
                digest=__import__('hashlib').sha256(r['text'].encode()).hexdigest()
                key=arm+'::'+rid+'::'+digest
                if key not in cached:
                    grade=score(questions[rid],r)
                    grade.update(truncated=r['finish_reason']=='length',
                        correct_completed=grade['answer_matches'] and r['finish_reason']!='length')
                    cached[key]=grade
                scored.append(dict(arm=arm,id=rid,dataset=questions[rid]['dataset'],finish_reason=r['finish_reason'],
                    tokens=r['generated_tokens'],output_sha256=digest,**cached[key]))
        write(cached_path,cached)
        with (ROOT/'artifacts/quality_raw.jsonl').open('w') as stream:
            for r in scored:stream.write(json.dumps(r)+'\n')
        summary=[]
        for arm in ARMS:
            for domain in ['gsm8k','math500']:
                selected=[r for r in scored if r['arm']==arm and r['dataset']==domain]
                if not selected:continue
                row=dict(arm=arm,dataset=domain,N=len(selected),correct=sum(r['answer_matches'] for r in selected),
                    accuracy_pct=100*np.mean([r['answer_matches'] for r in selected]),
                    correct_completed=sum(r['correct_completed'] for r in selected),
                    EOS=sum(r['finish_reason']=='eos' for r in selected),stop=sum(r['finish_reason']=='stop' for r in selected),
                    truncated=sum(r['truncated'] for r in selected),parse_fail=sum(not r['parsed'] for r in selected),
                    total_generated_tokens=sum(r['tokens'] for r in selected))
                for comparator in ['A0','A3']:
                    ref={r['id']:r for r in scored if r['arm']==comparator and r['dataset']==domain}
                    paired=[r for r in selected if r['id'] in ref]
                    if paired:row['vs_'+comparator]=paired_accuracy([r['answer_matches'] for r in paired],[ref[r['id']]['answer_matches'] for r in paired])
                summary.append(row)
        write(ROOT/'artifacts/quality_summary.json',summary)
        write_csv(ROOT/'artifacts/quality_summary.csv',summary)
        qp=[]
        for a,b in [('A0','A1'),('A0','A2'),('A0','A3'),('A0','A4'),('A1','A2'),('A3','A4')]:
            if quality[a] and quality[b]:qp.append(dict(left=a,right=b,**parity(list(quality[a].values()),list(quality[b].values()))))
        write(ROOT/'artifacts/quality_output_parity.json',qp)
        flips=[]
        lookup={(r['arm'],r['id']):r for r in scored}
        for a,b in [('A0','A1'),('A0','A2'),('A0','A3'),('A0','A4'),('A1','A2'),('A3','A4')]:
            for rid in sorted(set(quality[a])&set(quality[b])):
                left,right=lookup[a,rid],lookup[b,rid]
                if left['answer_matches']==right['answer_matches']:continue
                flips.append(dict(left_arm=a,right_arm=b,id=rid,dataset=left['dataset'],
                    direction='correct_to_wrong' if left['answer_matches'] else 'wrong_to_correct',
                    left_parsed_answer=left['extracted_answer'],right_parsed_answer=right['extracted_answer'],
                    gold=left['reference_answer'],left_tokens=left['tokens'],right_tokens=right['tokens'],
                    left_finish_reason=left['finish_reason'],right_finish_reason=right['finish_reason']))
        write(ROOT/'artifacts/quality_answer_flips.json',flips)
    print('ANALYZED',len(metrics),'cohorts;',len(points),'scaling cells')

if __name__=='__main__':main()

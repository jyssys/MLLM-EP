"""Process-block statistics. Raw kernel counts never become E2E speedups."""
import csv
import json
from collections import defaultdict
import numpy as np
from .common import ROOT,write,digest

def csv_write(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def interval(values,seed=428):
    values=np.array(values,float);rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return [float(x) for x in np.percentile(means,[2.5,97.5])]

def main():
    raw=ROOT/'raw';rows=[];rotation=[];output_pairs=[]
    for prefix in ('p2_screen_v2','p2_quiet_v1','cal_v1','p2_ablation_v1','p2_barrier_v1','baseline_mixed_quiet_v1','baseline_fair_mixed_quiet_v1','baseline_fair_refill_quiet_v1','baseline_mixed_groups_v1'):
        for folder in sorted(raw.glob(prefix+'*')):
            if not (folder/'completed.json').exists():continue
            identity=json.loads((folder/'identity.json').read_text())
            rotation.append(dict(run_id=folder.name,gpu=identity['gpu'],pid=identity['pid'],job=identity['job']))
            for path in sorted(folder.glob('timing*.json')):
                r=json.loads(path.read_text());data=r.pop('data',None)
                if data:r['decode_row_loop_work']=sum(max(0,len(q['tokens'])-1)*4 for q in data['requests'])
                if data:
                    output_pairs.append(dict(run=folder.name,P=r['P'],N=r['N'],D=r['D'],repeat=r['repeat'],
                        outputs={s['id']:dict(tokens=s['tokens'],depths=s['depths'],finish=s['finish']) for s in data['requests']}))
                r['run_id']=folder.name;r['path']=str(path.relative_to(ROOT));rows.append(r)
    csv_write('results_raw.csv',rows);write(ROOT/'GPU_policy_rotation.json',rotation)
    call_counts=[];ordered=defaultdict(list)
    for r in rows:ordered[(r['run_id'],r['P'],r['N'],r['D'])].append(r)
    for key,items in ordered.items():
        items.sort(key=lambda x:x['repeat'])
        for before,after in zip(items,items[1:]):
            replays=after['graph_replays']-before['graph_replays']
            call_counts.append(dict(run_id=key[0],P=key[1],N=key[2],D=key[3],repeat=after['repeat'],lane=after['lane'],
                graph_replays_in_cohort=replays,captures_in_cohort=after['graph_captures']-before['graph_captures'],
                fallbacks_in_cohort=after['graph_fallbacks']-before['graph_fallbacks'],
                decode_row_loop_work=after.get('decode_row_loop_work'),
                rows_per_replay=after.get('decode_row_loop_work',0)/replays if replays else None))
    csv_write('clean_graph_and_work_counts.csv',call_counts)
    byoutput=defaultdict(dict)
    for r in output_pairs:byoutput[(r['run'],r['P'],r['N'],r['repeat'])][r['D']]=r['outputs']
    parity=[]
    for (run,P,N,repeat),arms in byoutput.items():
        if 1 not in arms:continue
        for D,outputs in arms.items():
            if D==1:continue
            for rid,b in arms[1].items():
                a=outputs[rid]
                parity.append(dict(run_id=run,P=P,N=N,repeat=repeat,D=D,id=rid,
                    length_equal=len(a['tokens'])==len(b['tokens']),token_exact=a['tokens']==b['tokens'],
                    depth_exact=a['depths']==b['depths'],stop_equal=a['finish']==b['finish']))
    csv_write('bounded_output_parity.csv',parity)
    paired=[];byblock=defaultdict(dict)
    for r in rows:
        key=(r['run_id'],r['P'],r['N'])
        byblock[key].setdefault(r['D'],[]).append(r)
    grouped=defaultdict(list)
    for (run,P,N),groups in byblock.items():
        if 1 not in groups:continue
        baseline=np.mean([x['decode_tokens_s'] for x in groups[1]])
        for D,items in groups.items():
            rate=np.mean([x['decode_tokens_s'] for x in items])
            latency=np.mean([x['ITL_mean_ms'] for x in items])
            paired.append(dict(run_id=run,P=P,N=N,D=D,rate_speedup=rate/baseline,
                ITL_mean_ms=latency,decode_tokens_s=rate,baseline_tokens_s=baseline,
                lane=items[0]['lane'],repeats=len(items)))
            grouped[(items[0]['lane'],P,N,D)].append(paired[-1])
    csv_write('paired_process_blocks.csv',paired)
    summary=[]
    for (lane,P,N,D),items in sorted(grouped.items()):
        rates=[r['rate_speedup'] for r in items];lo,hi=interval(rates)
        summary.append(dict(lane=lane,P=P,N=N,D=D,process_blocks=len(items),
            throughput_speedup=float(np.mean(rates)),CI95_low=lo,CI95_high=hi,
            ITL_mean_ms=float(np.mean([r['ITL_mean_ms'] for r in items])),
            decode_tokens_s=float(np.mean([r['decode_tokens_s'] for r in items]))))
    csv_write('benchmark_summary.csv',summary)
    csv_write('baseline_results.csv',[r for r in summary if r['D']==1])
    comparisons=[]
    for folder in sorted(list(raw.glob('p1_numeric_v2*'))+list(raw.glob('p1_numeric_v3*'))):
        for path in sorted(folder.glob('numeric*.json')):
            r=json.loads(path.read_text())
            if 'comparison' in r:
                comparisons.append(dict(run_id=folder.name,P=r['P'],N=r['N'],D=r['D'],
                    graphs=r['graphs'],asynchronous=r['asynchronous'],**r['comparison']))
    csv_write('numerical_comparisons.csv',comparisons)
    write(ROOT/'numerical_summary.json',dict(comparisons=len(comparisons),
        token_exact=sum(r['tokens_exact'] for r in comparisons),KV_exact=sum(r['KV_exact'] for r in comparisons),
        numerical_pass=sum(r['numerical_pass'] for r in comparisons),
        max_relative_L2=max((r['max_relative_L2'] or 0 for r in comparisons),default=None)))
    frontier=[]
    for folder in sorted(raw.glob('p1_frontier_numeric_v1*')):
        for path in sorted(folder.glob('numeric*.json')):
            r=json.loads(path.read_text())
            if 'comparison' in r:frontier.append(dict(run_id=folder.name,P=r['P'],N=r['N'],D=r['D'],**r['comparison']))
    csv_write('frontier_numerical_comparisons.csv',frontier)
    stops=[]
    for folder in sorted(raw.glob('p1_natural_stop_v2*')):
        for path in sorted(folder.glob('natural_stop*.json')):
            r=json.loads(path.read_text());stops.append(dict(run_id=folder.name,D=r['D'],prompt=r['prompt'],
                output_tokens=len(r['output_ids']),stop=r['stop'],tokens_exact=r['token_exact_to_D1'],cache_released=r['cache_released']))
    csv_write('natural_stop_lifecycle.csv',stops)
    contention=defaultdict(dict)
    for r in rows:
        if r['run_id'].startswith('cal_v1'):
            contention[(r['gpu'],r['P'],r['N'])].setdefault(r['lane'],[]).append(r['decode_tokens_s'])
    drifts=[]
    for (gpu,P,N),lanes in sorted(contention.items()):
        if 'CALIBRATION_W1' not in lanes:continue
        ref=np.mean(lanes['CALIBRATION_W1'])
        for lane,values in lanes.items():
            drifts.append(dict(gpu=gpu,P=P,N=N,lane=lane,throughput_drift_pct=100*(np.mean(values)/ref-1),
                repeats=len(values),independent_starts=1))
    csv_write('contention_drift.csv',drifts)
    interventions=[]
    for folder in sorted(raw.glob('p2_intervene_v2*')):
        files=[json.loads(p.read_text()) for p in folder.glob('intervention*.json')]
        arms={r['action']:r for r in files}
        if not {'KEEP4','YIELD1'}<=set(arms):continue
        k,y=arms['KEEP4'],arms['YIELD1']
        item=dict(run_id=folder.name,P=k['P'],old=k['old'],new=k['new'],
            same_initial_state=k['state_sha256']==y['state_sha256'],
            outputs_exact=k['outputs']==y['outputs'],KEEP_all_ms=k['all_completion_ms'],YIELD_all_ms=y['all_completion_ms'],
            YIELD_all_speedup=k['all_completion_ms']/y['all_completion_ms'],
            KEEP_old_ms=k['old_completion_ms'],YIELD_old_ms=y['old_completion_ms'],
            KEEP_new_ms=k['new_completion_ms'],YIELD_new_ms=y['new_completion_ms'],evidence='LIVE_MEASURED_CONTROLLED_SYNC_COMPONENT')
        if 'NO_REFORM1' in arms:
            n=arms['NO_REFORM1'];item.update(NO_REFORM1_old_ms=n['old_completion_ms'],NO_REFORM1_all_ms=n['all_completion_ms'],
                KEEP4_vs_NO_REFORM1_old_speedup=n['old_completion_ms']/k['old_completion_ms'],
                no_reform_same_state=n['state_sha256']==k['state_sha256'],no_reform_output_exact=n['outputs']==k['outputs'])
        interventions.append(item)
    csv_write('paired_interventions.csv',interventions)
    profiles=[];observer_controls=[]
    for folder in sorted(raw.glob('p2_profile_v1*')):
        for file in folder.glob('profile*.json'):
            r=json.loads(file.read_text());groups=[q for q in r['events'] if q['spans'] and all(s['phase']=='recurrent' for s in q['spans'])]
            observer_controls.append(dict(P=r['P'],N=r['N'],D=r['D'],token_parity=r.get('observer_token_parity'),
                observed_wall_ratio=r.get('observer_wall_ratio'),scope='Single-start bracketed diagnostic, not pure overhead or clean serving.'))
            services=[q['service_ms'] for q in groups]
            cpums=[(q['end_ns']-q['start_ns'])/1e6 for q in r['scheduler']]
            loops=sum(s['G'] for q in groups for s in q['spans'])
            profiles.append(dict(P=r['P'],N=r['N'],D=r['D'],decode_submissions=len(groups),
                physical_row_loops=loops,physical_batch_loops=sum(q['spans'][0]['G'] for q in groups),
                graph_replays=sum(q['graph_after'].get('replays',0)-q['graph_before'].get('replays',0) for q in groups),
                decode_host_submit_ms=sum(q['host_dispatch_ms'] for q in groups),
                service_sum_ms=sum(services),scheduler_calls=len(cpums),scheduler_CPU_sum_ms=sum(cpums),
                scheduler_CPU_P95_ms=float(np.percentile(cpums,95)),
                additive_E2E_attribution='NOT_IDENTIFIED',evidence='HEAVY_OBSERVER'))
    csv_write('cpu_gpu_overhead.csv',profiles)
    csv_write('observer_control.csv',observer_controls)
    artifacts=[]
    for folder in sorted(raw.iterdir()):
        if not folder.is_dir():continue
        for p in sorted(folder.iterdir()):
            if p.is_file():artifacts.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)))
    write(ROOT/'raw_evidence_manifest.json',artifacts)
    print(json.dumps(dict(timing_rows=len(rows),blocks=len(byblock),numerical_pairs=len(comparisons),interventions=len(interventions),profiles=len(profiles))))
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,4))
        for D in (2,4):
            selected=[r for r in summary if r['lane']=='PARALLEL_CHARACTERIZATION' and r['D']==D]
            yy=np.array([r['throughput_speedup'] for r in selected]);lo=np.array([r['CI95_low'] for r in selected]);hi=np.array([r['CI95_high'] for r in selected])
            ax.errorbar(range(len(selected)),yy,yerr=[yy-lo,hi-yy],marker='o',capsize=2,label=f'D{D}/D1, process-block 95% CI')
        selected=[r for r in summary if r['lane']=='PARALLEL_CHARACTERIZATION' and r['D']==2]
        ax.set_xticks(range(len(selected)),[f"P{r['P']}/N{r['N']}" for r in selected],rotation=55,ha='right')
        ax.axhline(1,color='black',linewidth=.8);ax.axhline(1.03,color='gray',linestyle='--',label='Exploratory 3% screen')
        ax.set_ylabel('LIVE measured completed-token throughput speedup');ax.legend();fig.tight_layout()
        (ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/decode_group_speedup.png',dpi=160);plt.close(fig)
    if profiles:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        candidates=list(raw.glob('p2_profile_v1*/profile_P256_N1_D*.json'))
        fig,axes=plt.subplots(len(candidates),1,figsize=(12,2.4*len(candidates)),squeeze=False)
        for ax,path in zip(axes.flat,sorted(candidates)):
            record=json.loads(path.read_text());first=max(r['times'][0] for r in record['data']['requests'])
            origin=first
            for q in record['events']:
                start,end=q['aligned_service_begin_s'],q['aligned_service_end_s']
                if end<first or start>first+.16:continue
                phase=q['spans'][0]['phase'];color={'recurrent':'#2864ad','coda':'#dd7733','prelude':'#888888'}.get(phase,'#36a36a')
                ax.broken_barh([((start-origin)*1000,(end-start)*1000)],(.5,.5),facecolors=color)
                ax.broken_barh([((q['host_dispatch_begin_s']-origin)*1000,q['host_dispatch_ms'])],(1.3,.3),facecolors='#9c57a4')
            ax.set_title(f"LIVE event observer: P256/N1/D{record['D']}. GPU intervals include service gaps; host/GPU can overlap.")
            ax.set_yticks([.75,1.45],['GPU ready-stream interval','Host submission']);ax.set_xlim(0,160);ax.set_xlabel('ms after first delivered token')
        fig.tight_layout();fig.savefig(ROOT/'figures/measured_decode_timeline.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()

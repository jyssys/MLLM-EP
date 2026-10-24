"""Export measured rows, process-level uncertainty, and separate model bounds."""
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import ROOT,write

def csvfile(name,rows):
    path=ROOT/name;path.parent.mkdir(exist_ok=True,parents=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def confidence(values):
    values=np.asarray(values,float)
    rng=np.random.default_rng(20261024)
    medians=np.mean(rng.choice(values,(10000,len(values)),replace=True),1)
    return [float(np.percentile(medians,2.5)),float(np.percentile(medians,97.5))]

def main():
    results=[];memory=[];matrix=[];jobs={};cohorts={};memory_summary=[]
    for path in (ROOT/'jobs').glob('*.json'):
        job=json.loads(path.read_text());jobs[job['id']]=job
    for name,job in sorted(jobs.items()):
        directory=ROOT/'raw'/name
        done=directory/'completed.json';failed=directory/'failure.json'
        matrix.append(dict(id=name,kind=job['kind'],gpu=job['gpu'],P=job.get('P'),N=job.get('N'),
            status='COMPLETED' if done.exists() else 'FAILED' if failed.exists() else 'NOT_RUN',
            source_split=job.get('split'),graphs=job.get('graphs',True),sample_memory=job.get('sample_memory',True)))
        if not done.exists():continue
        if (directory/'memory_facts.json').exists():
            mf=json.loads((directory/'memory_facts.json').read_text());weights=json.loads((directory/'model_facts.json').read_text())['weight_bytes']
            timing=json.loads(done.read_text())['summary'].get('rows',[])
            samples=[]
            if (directory/'telemetry.jsonl').exists():
                for line in (directory/'telemetry.jsonl').read_text().splitlines():
                    sample=json.loads(line)
                    if 'stats' in sample:samples.append(float(sample['stats'].split(',')[2].strip()))
            peak=max([r['peak_bytes'] if 'peak_bytes' in r else r.get('peak_allocated',0) for r in timing]+[mf['allocated_after_engine']])
            memory_summary.append(dict(run=name,P=job.get('P'),N=job.get('N'),pool_bytes=mf['pool_bytes'],weight_bytes=weights,
                prompt_payload_bytes=mf['logical_prompt_payload_bytes'],allocated_after_engine=mf['allocated_after_engine'],
                reserved_after_engine=mf['reserved_after_engine'],peak_allocated=peak,
                peak_reserved=max([r.get('peak_reserved',0) for r in timing]+[mf['reserved_after_engine']]),
                non_KV_nonweight_peak_allocated=peak-weights-mf['pool_bytes'],
                NVML_sampled_peak_MiB=max(samples) if samples else None,
                evidence='LIVE_MEASURED_SERVING',qualifier='Payload is structural. NVML samples are not instantaneous peaks. Residual allocation is not exact graph workspace.'))
        for row in json.loads(done.read_text())['summary'].get('rows',[]):
            if 'ITL_mean_ms' not in row:continue
            results.append(dict(id=name,**row))
        for path in directory.glob('measured_*.json'):
            data=json.loads(path.read_text())['data']
            cohorts[(name,int(path.stem.split('_')[1]))]=data['requests']
            for index,event in enumerate(data['memory']):
                prefixes=[v['initialized_prefixes'] for v in event['allocations'].values()]
                memory.append(dict(run=name,index=index,host_s=event['host_s'],used_blocks=event['used_blocks'],
                    allocated=event['allocated_bytes'],reserved=event['reserved_bytes'],
                    live_requests=len(prefixes),
                    issued_valid_KV_bytes=sum(sum(sum(layer for layer in depth) for depth in request) for request in prefixes)*16*128*2*2,
                    evidence='LIVE_MEASURED_SERVING',
                    qualifier='Written-prefix metadata can precede GPU completion in Async mode. No claim of simultaneous initialized HBM traffic.'))
    for p in (1024,2048,4096,8192,16384,32768,65408):
        for n in (1,2,4,8):
            blocks=n*((p+128+15)//16)*4
            if blocks>18432:matrix.append(dict(id=f'capacity_skip_P{p}_N{n}',kind='timing',P=p,N=n,
                status='NOT_RUN',reason=f'{blocks} required blocks exceed registered 18432-block pool. Not an observed OOM.'))
    matrix.append(dict(id='exact_65536_prompt_plus_generation',kind='context_limit',P=65536,N=1,
        status='NOT_RUN',reason='Prompt plus output exceeds model position limit. Near-64K uses 65408 prompt tokens.'))
    csvfile('results_raw.csv',results);csvfile('memory_events.csv.gz',memory);csvfile('experiment_matrix.csv',matrix)
    model_completed=sum(r['status']=='COMPLETED' for r in matrix)
    model_failed=sum(r['status']=='FAILED' for r in matrix)
    csvfile('memory_summary.csv',memory_summary)
    grid=defaultdict(list)
    for r in results:
        if r['id'].startswith('grid_'):grid[(r['P'],r['N'],r['id'])].append(r)
    summary=defaultdict(list)
    for (p,n,name),rows in grid.items():
        summary[(p,n)].append(dict(id=name,ITL_ms=float(np.mean([r['ITL_mean_ms'] for r in rows])),
            TTFT_ms=float(np.mean([r['TTFT_mean_ms'] for r in rows])),
            peak_GiB=max(r['peak_allocated'] for r in rows)/2**30))
    baseline=[]
    for (p,n),rows in sorted(summary.items()):
        baseline.append(dict(P=p,N=n,starts=len(rows),ITL_mean_ms=float(np.mean([r['ITL_ms'] for r in rows])),
            TTFT_mean_ms=float(np.mean([r['TTFT_ms'] for r in rows])),
            logical_KV_GiB=p*n*786432/2**30,peak_GiB=max(r['peak_GiB'] for r in rows),
            evidence='LIVE_MEASURED_SERVING',lane='BOUNDED_HARDWARE_EOS_IGNORED',
            uncertainty='Two source-disjoint process starts. Technical repeats are not independent requests.'))
    csvfile('baseline_summary.csv',baseline)
    headroom=[]
    if (ROOT/'profile_complete_token_windows.json').exists():
        windows=json.loads((ROOT/'profile_complete_token_windows.json').read_text())
        for run in sorted({r['run'] for r in windows}):
            ws=[r for r in windows if r['run']==run and r['complete_R4_geometry']]
            if not ws:continue
            match=re.search(r'_P(\d+)_N(\d+)',run);p,n=map(int,match.groups())
            a=float(np.mean([r['attention_exclusive_GPU_service_ms'] for r in ws]));t=float(np.mean([r['output_completion_interval_ms'] for r in ws]));f=a/t
            b=next((r for r in baseline if (r['P'],r['N'])==(p,n)),None)
            item=dict(P=p,N=n,trace_attention_ms=a,trace_sampling_interval_ms=t,f_attention_trace=f,
                f_attention_clean='UNRESOLVED',logical_prompt_KV_GiB=p*n*786432/2**30,
                configured_pool_GiB=54,clean_grid_ITL_ms=b['ITL_mean_ms'] if b else None,
                peak_allocated_GiB=b['peak_GiB'] if b else None,
                max_completed_N_in_grid=max((r['N'] for r in baseline if r['P']==p),default=0),
                evidence='MODEL_UPPER_BOUND',scope='Conditional Amdahl model over measured heavy-trace GPU sampling intervals. Not measured clean E2E saving.')
            for speed in (1.5,2,4,float('inf')):
                key='infinite' if not np.isfinite(speed) else str(speed).replace('.','p')
                item[f'trace_reduction_fraction_at_{key}']=f*(1-1/speed)
            headroom.append(item)
    csvfile('headroom_bounds.csv',headroom)
    calibration=[]
    isolated=[r['ITL_mean_ms'] for r in results if r['id'].startswith('cal_w1_')]
    mean0=float(np.mean(isolated))
    for r in results:
        if not r['id'].startswith('cal_'):continue
        w=int(re.search(r'cal_w(\d+)',r['id'])[1])
        calibration.append(dict(run=r['id'],gpu=r['gpu'],workers=w,repeat=r['repeat'],
            ITL_ms=r['ITL_mean_ms'],relative_to_GPU0_isolated=r['ITL_mean_ms']/mean0,
            comparison='Interference comparison is paired on GPU0 only. Other ranks describe device spread.'))
    csvfile('contention_calibration.csv',calibration)
    dp=[]
    for width in (1,2,4,8):
        for start in (0,1):
            cohort=[]
            for gpu in range(width):
                path=ROOT/'raw'/f'cal_w{width}_r{start}_g{gpu}'/'measured_0.json'
                if path.exists():cohort.append(json.loads(path.read_text()))
            if len(cohort)!=width:continue
            begin=min(r['data']['start'] for r in cohort);end=max(r['data']['end'] for r in cohort)
            tokens=sum(len(q['tokens']) for r in cohort for q in r['data']['requests'])
            requests=sum(len(r['data']['requests']) for r in cohort)
            dp.append(dict(independent_replicas=width,start=start,requests=requests,tokens=tokens,
                common_cohort_makespan_s=end-begin,aggregate_output_tokens_s=tokens/(end-begin),
                peak_per_rank_GiB=max(r['peak_allocated'] for r in cohort)/2**30,
                evidence='LIVE_MEASURED_SERVING',scope='EOS-ignored bounded DP scale-out cohort. Not SLO serving capacity or B1 latency gain.'))
    csvfile('DP_replica_scaling.csv',dp)
    operator=[];communication=[];capacity=[]
    groups=list((ROOT/'distributed').glob('scaling_v2*'))+list((ROOT/'distributed').glob('scaling16_v1*'))+list((ROOT/'distributed').glob('pressure_v1*'))
    for group in sorted(groups):
        identities=list(group.glob('rank*_identity.json'))
        if not identities:continue
        world=json.loads(identities[0].read_text())['world']
        if not all((group/f'rank{r}_completed.json').exists() for r in range(world)):continue
        ranks=[json.loads((group/f'rank{r}_results.json').read_text()) for r in range(world)]
        assert len({len(rows) for rows in ranks})==1
        for rows in zip(*ranks):
            r=rows[0]
            assert all((x['fixture'],x['B'],x['placement'])==(r['fixture'],r['B'],r['placement']) for x in rows)
            if not all(x['graph'] is not None for x in rows):continue
            local=r['native']['graph'];samples=np.max([x['graph']['wall_ms'] for x in rows],axis=0)
            gpu_samples=np.max([x['graph']['cuda_ms'] for x in rows],axis=0)
            item=dict(run=group.name,P=world,L=r['L'],B=r['B'],depth=r['depth'],layer=r['layer'],placement=r['placement'],
                cache_pressure=bool(r.get('cache_pressure_bytes',0)),
                local_graph_wall_ms=local['wall_median_ms'],local_graph_cuda_ms=local['cuda_median_ms'],
                distributed_graph_wall_P50_ms=float(np.median(samples)),distributed_graph_wall_P95_ms=float(np.percentile(samples,95)),
                distributed_graph_cuda_P50_ms=float(np.median(gpu_samples)),
                speedup_wall=local['wall_median_ms']/float(np.median(samples)),
                speedup_cuda=local['cuda_median_ms']/float(np.median(gpu_samples)),
                max_relative_L2=max(max(x['numerical']['relative_L2'],x['graph']['numerical']['relative_L2']) for x in rows),
                max_absolute=max(max(x['numerical']['absolute_max'],x['graph']['numerical']['absolute_max']) for x in rows),
                native_capture_relative_L2=r['native']['original_capture_difference']['relative_L2'],
                byte_exact=all(x['GPU_shard_byte_exact'] for x in rows),split=r['split'],
                evidence='LIVE_MEASURED_OPERATOR',scope='Generic CP, model-derived replicated batch geometry. Not full-model E2E.')
            operator.append(item)
            logical=(world-1)*r['query_payload_bytes']+world*(world-1)*r['summary_payload_bytes']
            communication.append(dict(run=group.name,P=world,L=r['L'],B=r['B'],depth=r['depth'],layer=r['layer'],placement=r['placement'],
                query_bytes=r['query_payload_bytes'],summary_bytes=r['summary_payload_bytes'],
                application_delivered_bytes=logical,collectives_per_operator=2,
                full_R4_bytes_if_repeated_96_times=logical*96,
                full_R4_collectives_if_repeated_96_times=192,evidence='MODEL_UPPER_BOUND',
                qualifier='Tensor sizes and operator collectives are actual. Wire bytes are not measured. The 96-fold full-model repetition is structural only.'))
            capacity.append(dict(run=group.name,P=world,L=r['L'],B=r['B'],placement=r['placement'],
                max_rank_valid_operator_KV_bytes=max(x['owned_valid_KV_bytes'] for x in rows),
                max_rank_allocated_operator_KV_bytes=max(x['allocated_KV_bytes'] for x in rows),
                cluster_valid_operator_KV_bytes=sum(x['owned_valid_KV_bytes'] for x in rows),
                evidence='LIVE_MEASURED_OPERATOR',full_model_equivalent_max_rank_GiB=max(x['owned_valid_KV_bytes'] for x in rows)*96/2**30,
                full_model_equivalent_evidence='MODEL_UPPER_BOUND_NOT_FULL_CACHE_ALLOCATION'))
    csvfile('attention_operator_pairs.csv',operator);csvfile('shard_communication.csv',communication);csvfile('capacity_by_gpu_count.csv',capacity)
    pairgroups=defaultdict(list)
    for r in results:
        if r['id'].startswith('layout_'):
            match=re.match(r'layout_(native|layer_major)_P(\d+)_N(\d+)_s(\d+)(.*)',r['id'])
            if match:pairgroups[(int(match[2]),int(match[3]),int(match[4]),match[5],match[1])].append(r)
    pairs=[];parity=[]
    for (p,n,start,suffix,storage),values in pairgroups.items():
        if storage!='native':continue
        other=pairgroups.get((p,n,start,suffix,'layer_major'))
        if not other:continue
        a=float(np.mean([v['ITL_mean_ms'] for v in values]));b=float(np.mean([v['ITL_mean_ms'] for v in other]))
        pairs.append(dict(P=p,N=n,start=start,lane='QUIET' if suffix else 'PARALLEL',native_ITL_ms=a,layer_major_ITL_ms=b,
            speedup=a/b,
            native_cohort_E2E_ms=float(np.mean([v['E2E_cohort_ms'] for v in values])),
            layer_major_cohort_E2E_ms=float(np.mean([v['E2E_cohort_ms'] for v in other])),
            cohort_E2E_speedup=float(np.mean([v['E2E_cohort_ms'] for v in values]))/float(np.mean([v['E2E_cohort_ms'] for v in other])),
            TTFT_speedup=float(np.mean([v['TTFT_mean_ms'] for v in values]))/float(np.mean([v['TTFT_mean_ms'] for v in other])),
            evidence='LIVE_MEASURED_SERVING',scope='Exact generic layout, bounded generation, same physical GPU within pair.'))
        for native_row,candidate_row in zip(values,other):
            ar=cohorts[(native_row['id'],native_row['repeat'])];br=cohorts[(candidate_row['id'],candidate_row['repeat'])]
            assert len(ar)==len(br)
            for ra,rb in zip(sorted(ar,key=lambda x:x['id']),sorted(br,key=lambda x:x['id'])):
                first=next((i for i,(a,b) in enumerate(zip(ra['tokens'],rb['tokens'])) if a!=b),None)
                parity.append(dict(P=p,N=n,start=start,lane='QUIET' if suffix else 'PARALLEL',repeat=native_row['repeat'],
                    request=ra['id'],tokens_exact=ra['tokens']==rb['tokens'],lengths_exact=len(ra['tokens'])==len(rb['tokens']),
                    depths_exact=ra['depths']==rb['depths'],finish_exact=ra['finish']==rb['finish'],first_divergent_token=first,
                    native_tokens=len(ra['tokens']),candidate_tokens=len(rb['tokens']),evidence='LIVE_MEASURED_SERVING',scope='EOS-ignored hardware output parity. Not benchmark accuracy.'))
    csvfile('paired_comparisons.csv',pairs)
    csvfile('cohort_output_parity.csv',parity)
    aggregates=[]
    for key in sorted({(r['P'],r['L'],r['B'],r['placement'],r['cache_pressure']) for r in operator}):
        world,length,batch,placement,pressure=key
        perstart=defaultdict(list)
        for r in operator:
            if (r['P'],r['L'],r['B'],r['placement'],r['cache_pressure'])==key:perstart[r['run']].append(r)
        values=[float(np.mean([x['speedup_wall'] for x in rows])) for rows in perstart.values()]
        cuda=[float(np.mean([x['speedup_cuda'] for x in rows])) for rows in perstart.values()]
        aggregates.append(dict(P=world,L=length,B=batch,placement=placement,independent_starts=len(values),
            cache_pressure=pressure,
            speedup_wall=float(np.mean(values)),speedup_wall_95CI=confidence(values),speedup_cuda=float(np.mean(cuda)),
            local_wall_ms=float(np.mean([r['local_graph_wall_ms'] for rows in perstart.values() for r in rows])),
            distributed_wall_ms=float(np.mean([r['distributed_graph_wall_P50_ms'] for rows in perstart.values() for r in rows])),
            samples_per_start=len(next(iter(perstart.values()))),evidence='LIVE_MEASURED_OPERATOR'))
    write(ROOT/'operator_summary.json',aggregates)
    local=[]
    for path in (ROOT/'raw').glob('local_segmentation_s*/local_segmentation.json'):
        if not (path.parent/'completed.json').exists():continue
        for row in json.loads(path.read_text()):
            local.append(dict(run=path.parent.name,L=row['L'],B=row['B'],S=row['S'],cache_pressure=row['cache_pressure'],
                native_wall_ms=row['baseline']['wall_median_ms'],native_cuda_ms=row['baseline']['cuda_median_ms'],
                segmented_wall_ms=row['segmented']['wall_median_ms'],segmented_cuda_ms=row['segmented']['cuda_median_ms'],
                speedup_wall=row['baseline']['wall_median_ms']/row['segmented']['wall_median_ms'],
                relative_L2=row['numerical']['relative_L2'],absolute_max=row['numerical']['absolute_max'],evidence='LIVE_MEASURED_OPERATOR'))
    csvfile('local_segmentation.csv',local)
    strong=[]
    for length,batch,pressure in sorted({(r['L'],r['B'],r['cache_pressure']) for r in local}):
        choices=[]
        reference_rows=[dict(r,segmented_wall_ms=r['native_wall_ms']) for r in local
            if (r['L'],r['B'],r['cache_pressure'],r['S'])==(length,batch,pressure,2)]
        if reference_rows:choices.append((float(np.mean([r['segmented_wall_ms'] for r in reference_rows])),1,reference_rows))
        for split in (2,4,8,16):
            rs=[r for r in local if (r['L'],r['B'],r['cache_pressure'],r['S'])==(length,batch,pressure,split)]
            if rs:choices.append((float(np.mean([r['segmented_wall_ms'] for r in rs])),split,rs))
        if not choices:continue
        mean_local,split,ls=min(choices,key=lambda v:v[0])
        for world in (2,4,8):
            ds=[r for r in operator if (r['L'],r['B'],r['cache_pressure'],r['P'],r['placement'],r['depth'],r['layer'])==(length,batch,pressure,world,'balanced',1,12)]
            if not ds:continue
            local_values=np.array([r['segmented_wall_ms'] for r in ls]);dist_values=np.array([r['distributed_graph_wall_P50_ms'] for r in ds])
            rng=np.random.default_rng(20261024)
            ratios=np.mean(rng.choice(local_values,(10000,len(local_values)),replace=True),1)/np.mean(rng.choice(dist_values,(10000,len(dist_values)),replace=True),1)
            strong.append(dict(L=length,B=batch,cache_pressure=pressure,P=world,best_measured_local_S=split,
                local_starts=len(ls),distributed_starts=len(ds),local_wall_ms=mean_local,distributed_wall_ms=float(np.mean(dist_values)),
                speedup_over_best_local=mean_local/float(np.mean(dist_values)),CI_low=float(np.percentile(ratios,2.5)),CI_high=float(np.percentile(ratios,97.5)),
                evidence='LIVE_MEASURED_OPERATOR',qualifier='Independent-process bootstrap. Best S is a measured tuning envelope, not an online policy. Same representative model state.'))
    csvfile('best_local_vs_CP.csv',strong)
    for directory in sorted((ROOT/'distributed').iterdir()):
        if not directory.is_dir():continue
        identity=directory/'rank0_identity.json'
        if not identity.exists():continue
        world=json.loads(identity.read_text())['world']
        records=json.loads((directory/'rank0_results.json').read_text()) if (directory/'rank0_results.json').exists() else []
        matrix.append(dict(id=directory.name,kind='DISTRIBUTED_OPERATOR',gpu=','.join(map(str,range(world))),
            gpu_count=world,status='COMPLETED' if all((directory/f'rank{r}_completed.json').exists() for r in range(world)) else 'FAILED_PRESERVED',
            visible_key_lengths=','.join(map(str,sorted({r['L'] for r in records}))),
            batch_rows=','.join(map(str,sorted({r['B'] for r in records}))),
            evidence='LIVE_MEASURED_OPERATOR' if records else 'NOT_RUN'))
    csvfile('experiment_matrix.csv',matrix)
    write(ROOT/'analysis_counts.json',dict(completed_model_jobs=model_completed,
        failed_model_jobs=model_failed,timing_cohorts=len(results),
        operator_pairs=len(operator),baseline_shapes=len(baseline),layout_pairs=len(pairs)))

if __name__=='__main__':main()

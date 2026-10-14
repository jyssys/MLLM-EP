"""Paired clean effects and clock-anchored per-refinement activity."""
import collections
import gzip
import statistics
import numpy as np
from poc_dllm_multigpu_e2e_atlas.contract import *
from poc_dllm_multigpu_e2e_atlas.intervals import kernel_partition
from poc_dllm_multigpu_e2e_atlas.scripts.parse_timeline import classify,table
from poc_dllm_ep_comm.metrics import bootstrap_mean


def load(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def main():
    records=[];profile_rows=[];anchors=[];checks=[]
    for name in ['compact_calibration_p16384_v1','compact_calibration_p4096_v1','compact_heldout_p16384_v1']:
        directory=ROOT/'results/raw'/name
        if not (directory/'status.json').exists():continue
        selected=rows('held_out' if 'heldout' in name else 'calibration')
        for row in selected:
            rid=row['request_id'];clean={};coarse={}
            for arm in ['native','compact']:
                clean[arm]=[[load(directory/f'{rid}.{arm}.clean.{repeat}.rank{rank}.json.gz') for rank in range(4)] for repeat in range(3)]
                coarse[arm]=[load(directory/f'{rid}.{arm}.coarse.0.rank{rank}.json.gz') for rank in range(4)]
            truth=clean['native'][0][0]
            for arm in clean:
                for repeat,run in enumerate(clean[arm]):
                    for rank,value in enumerate(run):
                        assert value['NFE']==truth['NFE'] and value['output_ids']==truth['output_ids']
                        checks.append(dict(source=name,request_id=rid,arm=arm,repeat=repeat,rank=rank,pass_all=True))
            means={arm:statistics.mean(max(r['wall_ms'] for r in run) for run in clean[arm]) for arm in clean}
            record=dict(source=name,request_id=rid,domain=row['domain'],prefix=truth['prefix'],split=row['split'],
                native_ms=means['native'],compact_ms=means['compact'],reduction=1-means['compact']/means['native'],
                speedup=means['native']/means['compact'],NFE=truth['NFE'],output_tokens=truth['output_tokens'],
                native_TPS=truth['output_tokens']/(means['native']/1000),compact_TPS=truth['output_tokens']/(means['compact']/1000))
            for arm in clean:
                rank=max(range(4),key=lambda r:coarse[arm][r]['wall_ms'])
                spans=coarse[arm][rank]['observation']['spans']
                record[arm+'_warm_block_ms']=sum(s['GPU_interval_ms'] for s in spans if s['kind']=='block')
                record[arm+'_coarse_overhead']=max(r['wall_ms'] for r in coarse[arm])/means[arm]-1
                parity=json.loads((directory/f'parity.rank{rank}.json').read_text())
                assert all(p['output_acceptance_NFE_exact'] and p['regular_graph_logits_exact'] and all(p['regular_graph_KV_exact']) for p in parity)
            records.append(record)
        for arm in ['native','compact']:
            for rank in range(4):
                path=directory/f'torch_events.{arm}.rank{rank}.json.gz'
                if not path.exists():continue
                data=load(directory/f'coding_00.{arm}.profile.0.rank{rank}.json.gz')
                events=[e for e in load(path) if 'CUDA' in e['device_type']]
                marker=[e for e in events if 'FillFunctor<short' in e['name']]
                assert len(marker)==1,('Unique profiler clock anchor missing',arm,rank)
                span=next(s for s in data['observation']['spans'] if s['kind']=='profiler_clock_anchor')
                low=marker[0]['end_us']/1000-span['GPU_end_ms'];high=marker[0]['start_us']/1000-span['GPU_start_ms']
                assert high>=low-.005,'Incompatible CUPTI/CUDA-event clock anchors'
                offset=(low+high)/2;uncertainty=max(0,high-low)
                anchor_valid=uncertainty<=.1
                matching=[r for r in records if r['source']==name and r['request_id']=='coding_00'][0]
                clean_rank_mean=statistics.mean(run[rank]['wall_ms'] for run in
                    [[load(directory/f'coding_00.{arm}.clean.{rep}.rank{r}.json.gz') for r in range(4)] for rep in range(3)])
                overhead=data['wall_ms']/clean_rank_mean-1
                anchors.append(dict(source=name,arm=arm,rank=rank,offset_lower_ms=low,offset_upper_ms=high,
                    alignment_uncertainty_us=uncertainty*1000,anchor_valid=anchor_valid,
                    profile_overhead=overhead,profile_valid=overhead<=.10,
                    method='Distinct int16 fill enclosed by actual CUDA events. CUPTI event bounds constrain the offset.',
                    cross_rank_clock='Not aligned by this anchor. Use common-clock Nsight for four-rank Gantt.'))
                kernels=[]
                for event in events:
                    category=classify(event['name'])
                    if event['name']=='memcpy128':category='FULL_STORAGE_COPY' if arm=='native' else 'COMPACT_COPY'
                    kernels.append(dict(start=event['start_us']/1000,end=event['end_us']/1000,category=category))
                for scope in data['observation']['spans']:
                    if scope['kind'] not in ['refinement','commit_refresh']:continue
                    begin=scope['GPU_start_ms']+offset;end=scope['GPU_end_ms']+offset
                    grouped=collections.defaultdict(float)
                    for part in kernel_partition(kernels,begin,end):grouped[part['category']]+=part['end']-part['start']
                    for category,ms in grouped.items():profile_rows.append(dict(source=name,arm=arm,rank=rank,
                        block=scope['iteration']['block'],step=scope['iteration']['step'],cross=scope['iteration']['cross'],
                        category=category,exclusive_activity_ms=ms,fraction_of_refinement=ms/(end-begin),
                        refinement_interval_ms=end-begin,clock_uncertainty_us=uncertainty*1000,
                        validity='BOUNDED_EXCLUSIVE_RANK_ACTIVITY; full cross-rank dependency DAG remains PARTIAL',
                        profile_valid=overhead<=.10 and anchor_valid))
    summaries={}
    for name in sorted({r['source'] for r in records}):
        selected=[r for r in records if r['source']==name]
        numeric=[k for k,v in selected[0].items() if isinstance(v,(int,float)) and k!='prefix']
        summaries[name]={k:bootstrap_mean([r[k] for r in selected]) for k in numeric}
        summaries[name]['N']=len(selected)
    table(ROOT/'results/control_per_request.csv',records)
    table(ROOT/'results/per_refinement_profile_activity.csv',profile_rows)
    write(ROOT/'results/control_summary.json',dict(groups=summaries,checks=checks,
        no_cross_session_baseline_pooling=True,bootstrap='2000 paired request resamples',
        quality='NOT_RUN official benchmark. Exact bounded-output equivalence is a different result.',
        architecture='UNCHANGED TP4. Generic storage/layout control, not a new parallel architecture.'))
    write(ROOT/'results/profiler_clock_alignment.json',anchors)
    for name,v in summaries.items():print(name,'N',v['N'],'reduction%',v['reduction']['mean']*100,
        'CI',[x*100 for x in v['reduction']['ci95']])
    print('Clock anchor/profile validity',[(a['arm'],a['rank'],round(a['alignment_uncertainty_us'],2),round(a['profile_overhead']*100,2)) for a in anchors])


if __name__=='__main__':main()

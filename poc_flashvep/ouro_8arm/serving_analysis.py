"""Aggregate actual task timing separately from controlled hardware replay."""
import collections
import csv
import numpy as np
from .contract import *


def table(path,rows):
    if not rows:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    serving=[];memory=[];communication=[];exits=[];replays=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        values=[];cohorts=collections.defaultdict(list)
        if (folder/'setup.json').exists():
            setup=read(folder/'setup.json')
            for r in setup.get('ownership',[]):
                memory.append(dict(run=folder.name,cohort='native_setup',rank=r['rank'],physical_GPU=r['rank']+4,
                    weight_bytes=r.get('unique_parameter_storage_bytes'),KV_reserved_bytes=r.get('unique_KV_pool_storage_bytes'),
                    allocated_bytes=r.get('memory_allocated'),reserved_bytes=r.get('memory_reserved'),peak_bytes=r.get('max_memory_allocated'),
                    preparation_peak_bytes=None,
                    scope='Initialized native graph/setup peak only. Not the task-run peak.'))
        for file in sorted(folder.glob('*.json')):
            try:r=read(file)
            except json.JSONDecodeError:continue
            if isinstance(r,dict) and r.get('id') and 'output_ids' in r:values.append(r)
            if file.stem.startswith('cohort_'):
                cohorts[file.stem.split('_')[1]].append(r)
                rank=r.get('rank')
                if rank is not None:
                    m=r.get('memory',{});weights=m.get('weights',r.get('weights',{}))
                    memory.append(dict(run=folder.name,cohort=file.stem.split('_')[1],rank=rank,physical_GPU=rank+4,
                        weight_bytes=weights.get('total'),KV_reserved_bytes=m.get('KV_bytes',r.get('reserved_KV_bytes')),
                        allocated_bytes=m.get('allocated'),reserved_bytes=m.get('reserved'),peak_bytes=m.get('peak',r.get('peak')),
                        preparation_peak_bytes=m.get('preparation_peak_allocated',r.get('preparation_peak_allocated')),
                        scope='Peak is service-period PyTorch allocation. Preparation peak is separate when available. Neither value is a continuous NVML device peak.'))
                    counts=r.get('counts',{});bytes_=r.get('bytes',{})
                    for kind,count in counts.items():
                        if kind.startswith('peer_') or kind=='exit_KV_copies':
                            communication.append(dict(run=folder.name,cohort=file.stem.split('_')[1],rank=rank,kind=kind,
                                count=count,logical_bytes=bytes_.get(kind,bytes_.get('exit_KV') if kind=='exit_KV_copies' else None),
                                scope='Actual copy-call accounting. Not transport wire bytes or a bandwidth measurement.'))
                    if 'forward_loop_invocations' in r:
                        communication.append(dict(run=folder.name,cohort=file.stem.split('_')[1],rank=rank,kind='TP_row_all_reduce',
                            count=48*r['forward_loop_invocations'],logical_bytes=48*2048*4*r['token_loop_work'],
                            scope='Executed-loop-derived logical collective calls and FP32 tensor bytes per rank. Not wire bytes or independent timed API calls.'))
        if not values:continue
        lane=values[0]['lane'];arm=values[0]['arm']
        if lane=='FROZEN_TOKEN_HARDWARE_REPLAY':
            for index,rr in cohorts.items():
                controller=next((r for r in rr if r.get('rank')==0),None)
                if controller is None:continue
                elapsed=controller.get('makespan_s')
                if elapsed is None:elapsed=controller['end']-controller['origin']
                replays.append(dict(run=folder.name,arm=arm,case=controller['replay_case'],repeat=int(index),
                    makespan_s=elapsed,lane=controller['lane'],exclude_warmup=int(index)==0,
                    status='Numerically unresolved exploratory runtime.'))
            continue
        if not any(v.get('dataset') in ['gsm8k','math500'] for v in values):continue
        timing_valid=folder.name!='native_TP1_pilot_v1'
        all_count=len(values)
        if cohorts:
            finished={int(i) for i,rr in cohorts.items() if any(r.get('rank') in [0,None] for r in rr)}
            values=[r for r in values if r.get('cohort_index') in finished]
        if not values:continue
        total=0.;completed=0;resident=[]
        for index,rr in cohorts.items():
            controller=next((r for r in rr if r.get('rank') in [0,None]),None)
            if controller is None:continue
            duration=controller.get('makespan_s')
            if duration is None:duration=controller['end']-controller['origin']
            total+=duration;completed+=1
            if 'resident_model_cohort_wall_s' in controller:resident.append(controller['resident_model_cohort_wall_s'])
        if not cohorts and values[0].get('engine','').startswith('Pinned official CDB'):
            total=sum(r['E2E_s'] for r in values);completed=len(values)
        def mean(key):
            x=[r.get(key) for r in values if r.get(key) is not None]
            return float(np.mean(x)) if x else None
        serving.append(dict(run=folder.name,arm=arm,N=len(values),N_answer_records=all_count,run_complete=(folder/'complete.json').exists(),completed_cohorts=completed,
            cohort_makespan_sum_s=total if completed else None,mean_E2E_s=mean('E2E_s'),
            mean_TTFT_s=mean('TTFT_s') if timing_valid else None,mean_TPOT_s=mean('TPOT_s') if timing_valid else None,
            output_tokens=sum(r['output_tokens'] for r in values),
            output_tokens_per_s=sum(r['output_tokens'] for r in values)/total if total else None,
            requests_per_s=len(values)/total if total else None,mean_output_tokens=mean('output_tokens'),
            preparation_inclusive_cohorts=len(resident),preparation_inclusive_cohort_sum_s=sum(resident) if len(resident)==completed and completed else None,
            status=values[0]['status'],lane=lane,
            warning='Partial cohorts can have completed answers before cohort timing exists. Headline ratios require complete matched cohorts.'))
        depths=[d for r in values for d in r.get('depths',[])[1:]]
        if depths:
            counts=collections.Counter(depths);n=len(depths)
            exits.append(dict(run=folder.name,arm=arm,decode_tokens=n,R2=counts[2],R3=counts[3],R4=counts[4],
                mean_decode_depth=float(np.mean(depths)),actual_token_loop_work=sum(depths),fixed_R4_token_loop_work=4*n,
                skipped_loop_fraction=1-sum(depths)/(4*n),scope='Per-token executed depth accounting, not measured speedup.'))
    table(ROOT/'serving_by_arm.csv',serving);table(ROOT/'memory_by_rank.csv',memory)
    table(ROOT/'comm_profile.csv',communication);table(ROOT/'exit_trace_summary.csv',exits);table(ROOT/'replay_raw.csv',replays)
    print('AGGREGATED',len(serving),'task runs,',len(replays),'replays',flush=True)


if __name__=='__main__':main()

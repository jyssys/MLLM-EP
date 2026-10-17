"""Export measured summaries and figures with explicit timing scopes."""
import argparse
import collections
import gzip
import re
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_recurrent_asymmetry.contract import *
from poc_recurrent_asymmetry.scripts.analyze import table
from poc_recurrent_asymmetry.scripts.parse_trace import partition


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True);figures=args.output/'figures';figures.mkdir()
    a=args.analysis
    def save(name):
        plt.tight_layout();plt.savefig(figures/f'{name}.png',dpi=150);plt.savefig(figures/f'{name}.svg');plt.close()
    shapes=read(a/'profile_summary.json')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for B in [1,2,4,8]:
        rows=[r for r in shapes if r['batch']==B]
        for ax,field in zip(axes,['prefill_ms','cached_step_ms']):
            ax.plot([r['context'] for r in rows],[r[field] for r in rows],'-o',label=f'B{B}')
            ax.set(xlabel='Prompt tokens',ylabel='Measured wall ms',xscale='log',title=field);ax.legend()
    fig.suptitle('Exact HF reference, fixed cached inputs. Not natural-output throughput.');save('shape_latency')
    memory=list(__import__('csv').DictReader((a/'memory.csv').open()))
    plt.figure(figsize=(6,4))
    for B in [1,2,4,8]:
        rows=[r for r in memory if int(r['batch'])==B]
        plt.plot([int(r['context']) for r in rows],[float(r['logical_KV_GiB']) for r in rows],'-o',label=f'KV B{B}')
    plt.axhline(float(memory[0]['unique_weight_GiB']),color='k',ls='--',label='Unique tied weights')
    plt.xlabel('Prompt tokens');plt.ylabel('GiB, capacity not HBM traffic');plt.title('Audited exact R4 cache storage');plt.legend();save('memory_capacity')
    light=read(a/'light_components.json')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,P in zip(axes,[1024,16384]):
        rows=[r for r in light if r['P']==P and r['phase']=='decode']
        ax.bar([r['component'] for r in rows],[r['component_interval_ms'] for r in rows]);ax.tick_params(axis='x',rotation=25)
        ax.set(ylabel='ms per cached forward',title=f'P={P}, separate family runs')
    fig.suptitle('Single-family CUDA intervals. Bars do not form an exclusive E2E partition.');save('decode_components')
    cp=read(ROOT/'outputs/physical_v1/physical/cp_operator.json')
    plt.figure(figsize=(7,4))
    for name in ['native','CP4']:
        grouped=collections.defaultdict(list)
        for r in cp:grouped[r['context']].append(r[name]['median_ms'])
        plt.plot(sorted(grouped),[statistics.mean(grouped[P]) for P in sorted(grouped)],'-o',label=name)
    plt.ylabel('Physical operator host completion ms');plt.xlabel('Historical tokens');plt.title('Conventional exact CP microbenchmark, not E2E');plt.legend();save('CP_operator')
    p2p=read(ROOT/'outputs/physical_v1/physical/p2p.json')
    plt.figure(figsize=(7,4))
    grouped=collections.defaultdict(list)
    for r in p2p:grouped[r['payload_bytes']].append(r['median_ms'])
    plt.plot(sorted(grouped),[statistics.median(grouped[b]) for b in sorted(grouped)],'-o')
    plt.xscale('log');plt.yscale('log');plt.xlabel('Payload bytes');plt.ylabel('Median host completion ms over directed paths')
    plt.title('Actual P2P copies, GPUs 4–7. Includes launch/synchronization.');save('P2P_completion')
    # This partition uses actual correlated GPU events. It is not rescaled to clean wall time.
    trace=read(ROOT/'metrics/nsight/summary.json')['rows'];trace_rows=[]
    for r in trace:
        for category,duration in r['exclusive_ms'].items():
            trace_rows.append(dict(request=r['request'],phase=r['phase'],component=category,exclusive_ms=duration,
                fraction_of_instrumented_activity_envelope=duration/r['activity_envelope_ms'],
                validity='HEAVY_TRACE_UNVALIDATED_FOR_CLEAN_LATENCY'))
    table(args.output/'trace_exclusive.csv',trace_rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,P in zip(axes,[1024,16384]):
        rows=[r for r in trace if f'P={P}|' in r['request'] and 'DECODE' in r['phase']]
        sums=collections.defaultdict(float)
        for r in rows:
            for k,v in r['exclusive_ms'].items():sums[k]+=v/len(rows)
        names=sorted(sums,key=sums.get,reverse=True)[:6]
        ax.barh(names,[sums[k] for k in names]);ax.set(title=f'P={P}',xlabel='Instrumented interval ms')
    fig.suptitle('Heavy-trace local partition only. Observer overhead exceeds 10%.');save('trace_local_partition')
    loop_rows=[]
    for folder in sorted((ROOT/'outputs').glob('profile_P*_v1')):
        P=read(folder/'contract.json')['context']
        for phase in ['prefill','decode']:
            r=read(folder/f'{phase}_timeline.json')
            for loop in range(4):
                selected=[e for e in r['events'] if e['round']==loop]
                for category in sorted({e['category'] for e in selected}):
                    loop_rows.append(dict(P=P,batch=1,phase=phase,loop=loop,component=category,
                        inclusive_stream_interval_ms=sum(e['duration_ms'] for e in selected if e['category']==category),
                        observer_percent=r['observer_percent'],independent_requests=1,
                        validity='DIAGNOSTIC_ONLY. Nested intervals cannot be added. Not clean critical-path percent.'))
    table(args.output/'loop_components.csv',loop_rows)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,phase in zip(axes,['prefill','decode']):
        data=np.array([[sum(r['inclusive_stream_interval_ms'] for r in loop_rows
            if r['P']==P and r['phase']==phase and r['loop']==loop and r['component']=='LAYER')
            for loop in range(4)] for P in [1024,4096,8192,16384]])
        im=ax.imshow(data,aspect='auto');ax.set(xticks=range(4),xticklabels=['R1','R2','R3','R4'],
            yticks=range(4),yticklabels=['1K','4K','8K','16K'],title=phase)
        fig.colorbar(im,ax=ax,label='Diagnostic layer interval ms')
    fig.suptitle('Per-loop observer lane, one request/context. Not a clean-time decomposition.');save('loop_intervals')
    # Rank-local profiler clocks do not establish inter-rank start-time alignment.
    native_rows=[]
    for TP,version in [(1,'v1'),(2,'v1'),(4,'v3')]:
        folder=ROOT/f'outputs/native_TP{TP}_{version}'
        profile=read(folder/'kernel_profile.json')
        for path in sorted(folder.glob('torch_events.rank*.json.gz')):
            events=json.load(gzip.open(path,'rt'));rank=int(re.search(r'rank(\d+)',path.name).group(1))
            groups=collections.defaultdict(list)
            for event in events:
                if 'CUDA' not in event['device_type']:continue
                name=event['name'].lower()
                category=('TP_COLLECTIVE' if any(s in name for s in ['allreduce','all_reduce','cross_device_reduce','allgather','all_gather','reduce_scatter']) else
                    'ATTENTION' if 'flashattn' in name else 'KV_WRITE' if 'reshape_and_cache' in name else
                    'NORM' if 'rms_norm' in name else 'GEMM' if any(s in name for s in ['nvjet','gemm','gemv','ampere','sm90_xmma']) else 'OTHER')
                groups[category].append(event)
            for category,values in groups.items():
                native_rows.append(dict(TP=TP,rank=rank,category=category,kernels=len(values),
                    service_ms=sum(r['end_us']-r['start_us'] for r in values)/1000,
                    profile_observer_percent=profile['observer_percent'],
                    scope='Actual one 16K prefill and three cached forwards. Rank-local service, not global EP or E2E time.',
                    reference_qualification='REFERENCE_PARTIAL'))
    table(args.output/'native_kernel_service.csv',native_rows)
    native=read(a/'native_summary.json')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,field in zip(axes,['TTFT_ms','TPOT_ms']):
        for TP in [1,2,4]:
            rows=[r for r in native if r['TP']==TP and r['metric']==field]
            ax.plot([r['P'] for r in rows],[r['mean_request_median_ms'] for r in rows],'-o',label=f'TP{TP}')
        ax.set(xlabel='Prompt tokens',ylabel=field);ax.legend()
    fig.suptitle('Optimized native hardware controls. Cross-backend REFERENCE_PARTIAL.');save('native_TP_diagnostic')
    # Modeled timeline only. No physical pipeline was executed.
    tape=list(__import__('csv').DictReader((a/'modeled_timeline.csv').open()))
    fig,axes=plt.subplots(2,1,figsize=(12,5))
    for ax,placement in zip(axes,['layer_stationary_wrap','loop_stationary_replicated']):
        for r in tape:
            if r['placement']!=placement:continue
            start=float(r['start_ms']);end=float(r['end_ms']);device=int(r['device'])
            ax.broken_barh([(start,end-start)],(device-.35,.7),facecolors='tab:blue' if r['kind']=='COMPUTE' else 'tab:orange')
        ax.set(yticks=range(4),yticklabels=['GPU4','GPU5','GPU6','GPU7'],xlabel='Modeled ms',title=placement)
    fig.suptitle('SIMULATED legal schedule. Measured HF component and P2P inputs. Not a GPU timeline.');save('modeled_schedule')
    rec=read(ROOT/'outputs/profile_P1024_v1/recurrence_diagnostic.json')
    table(args.output/'recurrence_diagnostic.csv',rec)
    write(args.output/'manifest.json',dict(analysis=str(a),figures=[str(p.relative_to(args.output)) for p in sorted(figures.iterdir())],
        no_candidate_engine=True,heavy_trace_valid_for_clean_headline=False,
        scopes=['MEASURED_CLEAN_FIXED_STEP','SINGLE_FAMILY_STREAM_INTERVAL','HEAVY_TRACE_LOCAL','MODELED_SCHEDULE']))
    print('PRESENTATION',len(loop_rows),'loop rows',len(native_rows),'native service rows')


if __name__=='__main__':main()

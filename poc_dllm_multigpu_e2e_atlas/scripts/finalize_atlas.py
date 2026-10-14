"""Compact derived evidence and figures. Never turn service into E2E shares."""
import collections
import csv
import gzip
import statistics
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_dllm_multigpu_e2e_atlas.contract import *
from poc_dllm_multigpu_e2e_atlas.scripts.parse_timeline import table


def load(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def main():
    results=ROOT/'results';figs=ROOT/'figures';figs.mkdir(exist_ok=True)
    atlas=json.loads((results/'bootstrap_cis.json').read_text())['prefixes']
    control=json.loads((results/'control_summary.json').read_text())['groups']
    source=results/'raw/parsed_node_p16384_source_v1'
    kernels=load(source/'kernels.json.gz');copies=load(source/'copies.json.gz')
    layers=[];collectives=[];runtime=[]
    for rank in range(4):
        graph_copies=[c for c in copies if c['rank']==rank and not c['diagnostic'] and c['bytes']==335544320
            and c['scope'].startswith('ATLAS|graph_replay|')]
        by_graph=collections.defaultdict(list)
        for c in graph_copies:by_graph[c['correlation']].append(c)
        for correlation,values in by_graph.items():
            assert len(values)==40
            ordered=sorted(values,key=lambda r:r['start'])
            graph=[k for k in kernels if k['rank']==rank and not k['diagnostic'] and k['correlation']==correlation]
            for index,c in enumerate(ordered):layers.append(dict(rank=rank,correlation=correlation,layer=index//2,
                category='KV_D2D_COPY',service_ms=c['end']-c['start'],payload_bytes=c['bytes'],
                validity='AUX_HEAVY_TRACE_SOURCE_ORDER_MAPPING; not headline or full dependency critical time'))
            for category,count,first_layer in [('TP_ALL_REDUCE',40,0),('ROUTED_EXPERT_COMPUTE',38,1),('ATTENTION_COMPUTE',None,0)]:
                values=sorted([k for k in graph if k['category']==category],key=lambda r:r['start'])
                if count is None:
                    assert len(values) in [20,40];per=len(values)//20
                else:
                    assert len(values)==count,(rank,correlation,category,len(values));per=2
                for index,k in enumerate(values):layers.append(dict(rank=rank,correlation=correlation,
                    layer=index//per+first_layer,category=category,service_ms=k['end']-k['start'],payload_bytes='',
                    validity='AUX_HEAVY_TRACE_SOURCE_ORDER_MAPPING; not headline or full dependency critical time'))
            query=ordered[0]['iteration']['end']-ordered[0]['iteration']['start']
            collective=[k for k in graph if k['category']=='TP_ALL_REDUCE']
            collectives.append(dict(rank=rank,correlation=correlation,query_rows=query,count=len(collective),
                physical_type='NCCL AllReduce SUM BF16',input_payload_bytes_proxy_per_call=query*2048*2,
                service_ms=sum(k['end']-k['start'] for k in collective),
                payload_movement_ms='UNRESOLVED',rank_wait_ms='UNRESOLVED',EP_A2A_count=0,
                validity='Physical kernel type/count. Payload proxy is not wire bytes.'))
    apis=load(results/'raw/parsed_graph_p16384_v1/runtime.json.gz')
    for name in sorted({a['name'] for a in apis}):
        selected=[a for a in apis if a['name']==name]
        runtime.append(dict(name=name,count=len(selected),CPU_API_service_ms=sum(a['end']-a['start'] for a in selected),
            scope='All4 workers in one diagnostic request. CPU synchronization overlaps necessary GPU work, not removable overhead.'))
    table(results/'per_layer_or_graph_node.csv',layers);table(results/'per_rank_collectives.csv',collectives)
    table(results/'CPU_runtime_API_service.csv',runtime)
    external=[]
    for p in sorted(Path('/home/esjung/external/dllm-e2e-atlas-profiles').glob('*')):
        if p.is_file():external.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    write(results/'external_profile_inventory.json',external)
    held=control['compact_heldout_p16384_v1']
    write(results/'final_verdict.json',dict(verdict='GENERIC_ONLY',
        measured_B1_improvement_confirmed=True,new_parallel_architecture='NOT_ESTABLISHED',
        heldout_requests=held['N'],heldout_paired_reduction=held['reduction'],
        same_layout='TP4/EP1/DP1/PP1/CP1; physical4/5/6/7',
        physical_cause='slice_scatter preserves a monolithic backing storage, copying40x the logical layer KV view',
        exact_control='Independent compact out-of-place KV clone + fresh canvas write',
        full_critical_DAG='PARTIAL. Actual all-occurrence phase/iteration envelopes and source-correlated copy. Not a fictitious100% fine breakdown.',
        no_A2A=True,TP_PP_DP_alternatives='Analytical screening only. Candidate performance and workspace not physically validated.',
        quality='Official benchmark NOT_RUN. Exact output/acceptance/NFE/KV/logit parity on bounded requests PASS.',
        next_action='Promote the exact compact-copy reference, then measure any next architecture against it. Do not claim novelty from this generic correction.',
        branch='flashvep/dllm-multigpu-e2e-atlas'))

    def save(name):plt.tight_layout();plt.savefig(figs/name,dpi=160);plt.close()
    graph=results/'raw/parsed_graph_p16384_v1'
    events=load(graph/'kernels.json.gz')+load(graph/'copies.json.gz')
    mapping_path=results/'nsys_process_mapping.json'
    if not mapping_path.exists():
        conn=sqlite3.connect('/home/esjung/external/dllm-e2e-atlas-profiles/graph_p16384_v1.sqlite')
        mapping={str(rank):pid for rank,pid in conn.execute('select distinct deviceId,globalPid from CUPTI_ACTIVITY_KIND_KERNEL')}
        conn.close();write(mapping_path,mapping)
    mapping=json.loads(mapping_path.read_text())
    start=min(e['start'] for e in events);fig,ax=plt.subplots(figsize=(13,5))
    for rank in range(4):
        for e in [e for e in events if e['rank']==rank]:
            color='#377eb8' if e['name']=='NATIVE_CUDA_GRAPH_OPAQUE' else '#ff7f00'
            ax.broken_barh([(e['start']-start,e['end']-e['start'])],(rank-.32,.42),facecolors=color)
        for a in [a for a in apis if a['globalPid']==mapping[str(rank)] and 'GraphLaunch' in a['name']]:
            ax.broken_barh([(a['start']-start,a['end']-a['start'])],(rank+.16,.10),facecolors='#4daf4a')
    ax.set(yticks=range(4),yticklabels=[f'GPU{g}' for g in GPUS],xlabel='Common Nsight capture clock, relative ms',
        title='Actual4-GPU request timeline: blue native graph, orange other GPU work, green host launch\nGraph internals are opaque. Blank GPU gaps are not proven removable host delay.')
    save('four_rank_timeline.png')
    plt.figure(figsize=(10,4))
    for rank in range(4):
        local=[e for e in events if e['rank']==rank]
        begin=min(e['start'] for e in local);end=max(e['end'] for e in local)
        from poc_dllm_multigpu_e2e_atlas.intervals import kernel_partition
        for part in kernel_partition(local,begin,end):
            if part['active_count']==0:plt.broken_barh([(part['start']-start,part['end']-part['start'])],(rank-.3,.6),facecolors='#e41a1c')
    plt.yticks(range(4),[f'GPU{g}' for g in GPUS]);plt.xlabel('Common capture clock, relative ms')
    plt.title('Observed between-activity idle gaps only\nInternal graph idle and rank-wait cause remain unresolved');save('sync_idle_gaps.png')
    with (results/'per_refinement.csv').open() as stream:refs=list(csv.DictReader(stream))
    chosen=[r for r in refs if r['request_id']=='document_01' and r['prefix']=='16384' and r['repeat']=='0']
    matrix=np.full((2,max(int(r['step']) for r in chosen)+1),np.nan)
    for r in chosen:matrix[int(r['block']),int(r['step'])]=float(r['GPU_interval_ms'])
    plt.figure(figsize=(12,3));plt.imshow(matrix,aspect='auto');plt.colorbar(label='Actual GPU interval ms')
    plt.yticks([0,1],['Block0','Block1']);plt.xlabel('Actual iteration index, including refresh')
    plt.title('Native document_01 P16384: block/refinement intervals, no last-replay multiplication');save('block_refinement_heatmap.png')
    cats=['KV_D2D_COPY','ATTENTION_COMPUTE','ROUTED_EXPERT_COMPUTE','TP_ALL_REDUCE'];matrix=np.full((20,4),np.nan)
    for lid in range(20):
        for col,cat in enumerate(cats):
            values=[r['service_ms'] for r in layers if r['rank']==0 and r['layer']==lid and r['category']==cat]
            count=len({r['correlation'] for r in layers if r['rank']==0 and r['layer']==lid and r['category']==cat})
            if values:matrix[lid,col]=sum(values)/count
    colors=plt.get_cmap('viridis').copy();colors.set_bad('#bdbdbd')
    plt.figure(figsize=(8,7));plt.imshow(matrix,aspect='auto',cmap=colors);plt.colorbar(label='AUX service ms/layer/forward')
    plt.xticks(range(4),['KV D2D','Attention','Routed GEMM','TP AllReduce'],rotation=15);plt.yticks(range(20));plt.ylabel('Source-order layer')
    plt.title('Native graph nodes: heavy diagnostic service, NOT E2E critical share\nExtra eager forward excluded. Gray dense-layer expert cell is unmapped.');save('layer_category_heatmap.png')
    x=np.arange(len(PREFIXES));keys=[str(p) for p in PREFIXES]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for field,label in [('clean_E2E_ms','Clean E2E'),('prefill_GPU_interval_ms','Coarse prefill envelope'),('coarse_block_span_ms','Coarse warm block envelope')]:
        values=[atlas[k][field]['mean'] for k in keys]
        axes[0].plot(x,values,'o-',label=label)
    axes[0].set(xticks=x,xticklabels=keys,ylabel='ms',xlabel='Logical prefix');axes[0].legend(fontsize=8)
    axes[1].bar(x,[atlas[k]['outside_graph_optimistic_fraction']['mean']*100 for k in keys])
    axes[1].set(xticks=x,xticklabels=keys,ylabel='Upper bound (%)',xlabel='Logical prefix',title='Remove ALL outside-graph work: optimistic ceiling')
    fig.suptitle('New native session only. P512 coarse attribution UNTRUSTED.');save('prefix_scaling.png')
    profile=list(csv.DictReader((results/'per_refinement_profile_activity.csv').open()))
    categories=sorted({r['category'] for r in profile});fig,axes=plt.subplots(1,2,figsize=(14,4),sharey=True)
    for axis,arm in zip(axes,['native','compact']):
        selected=[r for r in profile if r['arm']==arm and r['rank']=='0'];steps=sorted({(int(r['block']),int(r['step'])) for r in selected})
        bottom=np.zeros(len(steps))
        for cat in categories:
            heights=[sum(float(r['exclusive_activity_ms']) for r in selected if r['category']==cat and (int(r['block']),int(r['step']))==step) for step in steps]
            axis.bar(range(len(steps)),heights,bottom=bottom,label=cat);bottom+=heights
        axis.set(title=arm,xlabel='Observed iteration',ylabel='Rank0 activity interval ms')
    axes[1].legend(fontsize=5,bbox_to_anchor=(1.02,1));fig.suptitle('Measured activity, NOT complete critical DAG\nNative clock bracket108–126us fails100us point gate. Preserve bounded attribution.');save('per_iteration_wall_budget.png')
    with (results/'control_per_request.csv').open() as stream:rows_control=list(csv.DictReader(stream))
    held_rows=[r for r in rows_control if r['split']=='held_out'];plt.figure(figsize=(6,5))
    a=[float(r['native_ms']) for r in held_rows];b=[float(r['compact_ms']) for r in held_rows]
    plt.scatter(a,b);plt.plot([min(a),max(a)],[min(a),max(a)],'k--');plt.xlabel('Native clean E2E ms');plt.ylabel('Exact compact clean E2E ms')
    plt.title('24 frozen held-out requests, paired3 repeats\nNo approximation or parallel-layout change');save('paired_e2e_control.png')
    with (results/'architecture_screening.csv').open() as stream:architectures=list(csv.DictReader(stream))
    fig,ax=plt.subplots(figsize=(12,3));ax.axis('off')
    cells=[[r['layout'],f"{float(r['weight_GiB_per_rank_proxy']):.2f}",r['status'],r['single_wave_PP_util_B1']] for r in architectures]
    tab=ax.table(cellText=cells,colLabels=['Layout','Weight GiB/rank proxy','Runtime evidence','Ideal B1 wave PP util'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.6)
    ax.set_title('Capacity/dependency screening, not a speedup prediction\nAlternative graph workspace and performance NOT_RUN');save('architecture_tradeoff.png')
    print('Figures, source-order node tables, collective counters, external hashes and final verdict written.')


if __name__=='__main__':main()

"""Draw controlled discovery and measured Host observations without fake kernels."""
import collections
from .common import *
from .analyze import records
from .final_analysis import load


def observations(name,ids):
    folder=ROOT/'raw'/name
    requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl') if 'q'+str(r['sample_id']) in ids}
    source={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl') if r['request_id'] in ids}
    starts={};imports={}
    for path in folder.glob('native_prefill_*.json'):
        for row in json.loads(path.read_text())['prompt_spans']:
            rid=row['request_id'].split('-')[0]
            if rid in ids:starts[rid]=min(starts.get(rid,float('inf')),row['time'])
    for path in folder.glob('extension_decode_*.jsonl'):
        for row in records(path):
            event=row['event']
            if event=='NATIVE_KV_COHORT_READY':
                for request in row['requests']:
                    rid=request.split('-')[0]
                    if rid in ids:imports[rid]=max(imports.get(rid,0.),row['time'])
            elif event=='HIDDEN_GPU_IMPORTED':
                rid=row['request_id'].split('-')[0]
                if rid in ids:imports[rid]=max(imports.get(rid,0.),row['time'])
    return dict(name=name,requests=requests,source=source,starts=starts,imports=imports)


def timeline(plt):
    runs=load()
    dynamic=next((r for r in runs if r['name'].startswith('heldout_capacity_Dynamic_')),None)
    if dynamic is None:return dict(status='NOT_RUN')
    native=next((r for r in runs if r['method']=='Native' and r['seed']==dynamic['seed'] and r['rate']==dynamic['rate']),None)
    if native is None:return dict(status='NOT_RUN_NO_SAME_RATE_REFERENCE')
    folder=ROOT/'raw'/dynamic['name'];begin=json.loads((folder/'measurement_begin.json').read_text())['time']
    rows=records(folder/'requests.jsonl')
    eligible=sorted([r for r in rows if r['arrival']>=begin+180 and r.get('completed_D_prefill_layers',0)>0],key=lambda r:r['sample_id'])
    if not eligible:return dict(status='NOT_RUN_NO_MIGRATED_REFERENCE_REQUEST')
    first=eligible[0]['sample_id'];ids=['q'+str(i) for i in range(first,first+6)]
    groups=[observations(r['name'],ids) for r in (native,dynamic)]
    assert all(set(g['requests'])==set(ids) for g in groups)
    for rid in ids:
        assert groups[0]['requests'][rid]['input_ids']==groups[1]['requests'][rid]['input_ids']
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
    colors=dict(queue='#bdbdbd',source='#e69f00',handoff='#56b4e9',suffix='#cc79a7',decode='#009e73')
    extracted=[]
    for ax,group in zip(axes,groups):
        origin=min(r['arrival'] for r in group['requests'].values());used=set()
        def bar(start,end,y,label,key):
            if end<start-1e-5:return
            ax.broken_barh([(start-origin,max(0.,end-start))],(y-.32,.64),facecolors=colors[key],
                label=label if label not in used else None);used.add(label)
        marks=[]
        for j,rid in enumerate(ids):
            row=group['requests'][rid];start=group['starts'][rid];ready=group['source'][rid]
            imported=group['imports'].get(rid)
            bar(row['arrival'],start,j,'P wait: admission + queue','queue')
            bar(start,ready,j,'P scheduling to Source-ready: Host observation','source')
            ax.scatter([ready-origin],[j],marker='|',s=85,color='black')
            if imported is not None:
                bar(ready,imported,j+7,'Handoff + admission + import: Host observation','handoff')
                bar(imported,row['first_token'],j+7,'Import-ready to first output: Host observation','suffix')
            else:
                bar(ready,row['first_token'],j+7,'Handoff + admission + import: Host observation','handoff')
            bar(row['first_token'],row['finish'],j+7,'Decode delivery interval: Host observation','decode')
            ax.scatter([row['first_token']-origin],[j+7],marker='*',s=50,color='black')
            marks.append(dict(request_id=rid,arrival=row['arrival'],P_scheduled=start,Source_ready=ready,
                D_import_ready=imported,first_output=row['first_token'],finish=row['finish'],
                completed_D_layers=row.get('completed_D_prefill_layers',0)))
        P_labels=[];D_labels=[]
        for rid in ids:
            moved=group['requests'][rid].get('completed_D_prefill_layers',0)
            P_labels.append(f'P {rid}: L1-{36-moved}')
            D_labels.append(f'D {rid}: L{37-moved}-36 + Decode' if moved else f'D {rid}: Decode')
        ax.set_yticks(list(range(6))+list(range(7,13)),P_labels+D_labels,fontsize=7)
        ax.set_title(group['name'],fontsize=8);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        extracted.append(dict(name=group['name'],origin=origin,requests=marks))
    axes[0].legend(fontsize=7,ncol=2,loc='upper right');axes[-1].set_xlabel('Seconds from the same selected request arrival in each independent run')
    fig.suptitle('Matched arrivals: Host observations, not CUDA kernel timelines. Vertical mark: Source-ready. Star: first output.')
    fig.tight_layout();fig.savefig(ROOT/'figures/final_Host_P_D_timeline.svg');plt.close(fig)
    return dict(status='MEASURED',runs=extracted,
        request_selection='First migrated sample in the first Dynamic confirmation start with arrival at least180 seconds after the run anchor. Include that sample and its next5 IDs. No latency ranking.',
        scope='Wait includes admission. P observation includes launch and scheduling gaps. Handoff includes queueing and import, not pure wire time. Import-to-output includes suffix, finalization and delivery. Decode uses delivery timestamps. No physical idle, exclusive kernel duration, or kernel overlap is inferred.')


def heterogeneous(plt):
    results=[];fig,axes=plt.subplots(2,2,figsize=(12,7))
    labels=('Native','Static8','Static12','Dynamic','Colocated');colors=('#666666','#56b4e9','#009e73','#d55e00','#0072b2')
    for line,kind in enumerate(('BURSTY','MIXED_OUTPUT')):
        for i,(label,color) in enumerate(zip(labels,colors)):
            suffix='_recovery_v2' if label=='Dynamic' else '_v1'
            stem=f'async_updated_{kind}_{label}_N{256 if kind=="BURSTY" else 128}_q{2.0 if kind=="BURSTY" else 1.0}{suffix}'
            path=ROOT/'analysis'/(stem+'_metrics.json')
            if not path.exists():continue
            value=json.loads(path.read_text());results.append(value['name'])
            axes[line,0].bar(i,value['TTFT_p95_ms']/1000,color=color)
            axes[line,1].bar(i,value['SLO_regimes']['Decode_priority']['Output_token_SLO_goodput'],color=color)
        for ax in axes[line]:ax.set_xticks(range(5),labels);ax.set_title(kind);ax.grid(axis='y',alpha=.2)
        axes[line,0].set_ylabel('TTFT p95 (s)');axes[line,1].set_ylabel('Decode-priority qualified tokens/s')
    fig.suptitle('Controlled discovery: one finite cohort per method. Not sustainable capacity or held-out evidence.')
    fig.tight_layout();fig.savefig(ROOT/'figures/final_7_bursty_and_mixed_output.svg');plt.close(fig)
    return results


def matched(plt):
    names={'Dynamic':'async_D_DIRECT_nominal_N128_O16_q4.8_v1',
        'Static same requests':'async_D_STATIC_matched_same_request_fraction_N128_O16_q4.8_v1',
        'Static same work':'async_D_STATIC_matched_same_layer_work_N128_O16_q4.8_v1',
        'Static same histogram':'async_D_STATIC_matched_same_cut_histogram_N128_O16_q4.8_v1',
        'Strong Static12':'async_D_STATIC_L12_N128_O16_q4.8_v1'}
    fig,axes=plt.subplots(1,2,figsize=(12,4));used=[]
    for label,name in names.items():
        path=ROOT/'analysis'/(name+'_metrics.json')
        if not path.exists():continue
        v=json.loads(path.read_text());used.append(name)
        axes[0].scatter(v['TTFT_p95_ms']/1000,v['ITL_p99_ms'],label=label,s=50)
        axes[1].scatter(v['TTFT_p95_ms']/1000,v['SLO_regimes']['Decode_priority']['Output_token_SLO_goodput'],label=label,s=50)
    for ax in axes:ax.set_xlabel('TTFT p95 (s)');ax.grid(alpha=.2)
    axes[0].set_ylabel('ITL p99 (ms)');axes[1].set_ylabel('Qualified output tokens/s');axes[0].legend(fontsize=7)
    fig.suptitle('N128 matched-budget discovery. Actual geometry differs. One start gives no independent policy claim.')
    fig.tight_layout();fig.savefig(ROOT/'figures/final_8_static_Dynamic_Pareto.svg');plt.close(fig)
    return used


def main():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    value=dict(time=time.time(),timeline=timeline(plt),heterogeneous_runs=heterogeneous(plt),matched_runs=matched(plt),
        physical_kernel_activity='NOT_MEASURED',evidence='MEASURED',
        scope='All figures use observed records. No conceptual GPU activity or future work is drawn as measured execution.')
    write(ROOT/'analysis/final_Host_timeline_and_controlled_figure_provenance.json',value)
    print('HOST_AND_CONTROLLED_FIGURES_COMPLETE',flush=True)


if __name__=='__main__':main()

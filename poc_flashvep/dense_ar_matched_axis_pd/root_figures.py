"""Plot separate evidence tiers for work, operator models and host paths."""
import json
import math
import statistics
from .common import ROOT,write
from .host_paths import records


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=[]
    def save(fig,stem,evidence,scope):
        for suffix in ('.png','.pdf'):
            fig.savefig(ROOT/'figures'/(stem+suffix),dpi=170,bbox_inches='tight')
        plt.close(fig)
        manifest.append(dict(stem=stem,evidence=evidence,scope=scope))
    path=ROOT/'analysis/matched_axis_structural_work.json'
    if path.exists():
        pairs=json.loads(path.read_text())['rows']
        added=ROOT/'analysis/last2_structural_work.json'
        if added.exists() and not (ROOT/'state/last2_supplement_cancelled.json').exists():
            pairs=sorted(pairs+json.loads(added.read_text())['rows'],key=lambda r:r['layers'])
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,role in zip(axes,('P','D')):
            for i,row in enumerate(pairs):
                for j,(axis,color) in enumerate((('layer','#0072b2'),('token','#8c56a0'))):
                    value=row['totals'][f'{axis}_{role}_attention_pairs']/144/1e9
                    ax.bar(i+(j-.5)*.32,value,width=.3,color=color,label=axis.title() if i==0 else None)
            ax.set_xticks(range(len(pairs)),[f'Last{r["layers"]}' for r in pairs])
            ax.set(title=role+'-side causal Attention work',ylabel='Billion causal pairs per request')
            ax.legend();ax.grid(axis='y',alpha=.2)
        fig.suptitle('STRUCTURAL_ACCOUNTING | Matched Transformer GEMM rows, different Attention distribution')
        fig.text(.02,.01,'Exact total over N144. No GPU-time or bandwidth inference. Full-system Attention work is equal across the two axes.',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1))
        save(fig,'root_attention_work_distribution','STRUCTURAL_ACCOUNTING','Exact causal pair counts, not GPU durations.')
    micro=ROOT/'raw/isolated_BF16_axis_geometry_v1/operator_geometry.json'
    if micro.exists():
        values=json.loads(micro.read_text())['results']
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        names=sorted({r['projection'] for r in values if r['kind']=='GEMM'})
        for ax,name in zip(axes.flat,names):
            for device,color in ((0,'#009e73'),(1,'#d95f02')):
                rows=sorted([r for r in values if r['kind']=='GEMM' and r['projection']==name and r['GPU']==device],key=lambda r:r['M'])
                ax.plot([r['M'] for r in rows],[r['enclosure_effective_TFLOPs_s'] for r in rows],marker='o',ms=3,color=color,label=f'GPU{device}')
            ax.set(title=name.split('layers.0.')[-1],xlabel='GEMM M rows',ylabel='FLOPs / event enclosure (TFLOP/s)')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('MEASURED isolated BF16 operators | Original layer0 weights | No serving speedup claim')
        fig.text(.02,.01,'One isolated process. GPU0 and GPU1 measured sequentially. 30 event repetitions per shape. No independent-start CI. Not an SM/Tensor Core counter.',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1))
        save(fig,'root_GEMM_shape_efficiency','MEASURED','Isolated event enclosures, not exclusive GPU compute or online speedup.')
    root=ROOT/'analysis/root_cause_components.json'
    if root.exists():
        data=json.loads(root.read_text())
        starts=data['host_starts']
        components=[('arrival_to_P_host_submit','P external wait','#a6cee3'),
            ('P_submit_to_source_notification','Source in-flight enclosure','#1f78b4'),
            ('source_notification_to_D_host_submit','D pending/admission wait','#b2df8a'),
            ('D_submit_to_first_worker_notice','D preparation enclosure','#6a3d9a'),
            ('first_worker_notice_to_client','Delivery','#ff7f00')]
        for m in (4,8,12):
            selected=[]
            fig,axes=plt.subplots(1,2,figsize=(12,4.5))
            for method,shift in (('Layer',-.16),('Token',.16)):
                host=[];decode=[];used=[]
                for cap in (16,32,64,128,200):
                    rows=[r for r in starts if r['method']==method and r['layers']==m and r['cap']==cap and r['QPS']==6 and r['count']==144 and r['admission']==8]
                    confirmation=[r for r in rows if r['stage']=='confirmation']
                    rows=confirmation if len(confirmation)>=3 else [r for r in rows if r['stage']=='discovery']
                    used.extend(r['name'] for r in rows)
                    host.append({key:statistics.mean(r['host_path_mean_ms'][key] for r in rows)/1000 if rows else math.nan for key,_,_ in components})
                    decode.append(statistics.mean(r['summed_decode_residency_per_request_ms'] for r in rows) if rows else math.nan)
                bottom=[0.]*5
                for key,label,color in components:
                    y=[r[key] for r in host]
                    axes[0].bar([i+shift for i in range(5)],y,bottom=bottom,width=.3,color=color,label=label if method=='Layer' else None)
                    bottom=[a+b for a,b in zip(bottom,y)]
                axes[1].plot((16,32,64,128,200),decode,marker='o',label=method)
                selected.extend(used)
            if selected:
                axes[0].set_xticks(range(5),['16','32','64','128','200'])
                axes[0].set(xlabel='Output tokens (Layer left, Token right)',ylabel='Mean TTFT host identity (s)')
                axes[0].legend(fontsize=7)
                axes[1].set(xlabel='Output tokens',ylabel='Summed cohort residency / completed request (ms)')
                axes[1].legend();axes[1].grid(alpha=.2)
                fig.suptitle(f'TRACE_ANALYSIS | Last{m} matched work | QPS6 | B8192/S1024, PD cap8')
                fig.text(.02,.01,'N144. Discovery n=1 or complete confirmation n=3. Means only. No additive p95 attribution or exclusive-kernel interpretation. Residency uses worker notices.',fontsize=8)
                fig.tight_layout(rect=(0,.04,1,1))
                save(fig,f'root_host_critical_path_Last{m}','TRACE_ANALYSIS',dict(names=selected,stage_statistic='Mean of per-start host means. No CI on stacked component identities.'))
            else:
                plt.close(fig)
    gates=ROOT/'analysis/online_profiler_observer_gates.json'
    if gates.exists():
        from .kernel_analysis import analyze
        for row in json.loads(gates.read_text()):
            kernel=analyze(row['names'][1],row['passed'])
            timeline(plt,row,kernel,save)
    safe=ROOT/'analysis/safe_observer_API_placement.json'
    if safe.exists():
        from .kernel_analysis import analyze
        record=json.loads(safe.read_text())
        row=dict(names=record['names'],method='Layer',cap=16,passed=False,
            figure_suffix='_safe_API',scope='Profiler API calls occur outside the measured window. The full observer gate did not pass.')
        timeline(plt,row,analyze(row['names'][1],False),save)
    write(ROOT/'analysis/root_figure_manifest.json',dict(figures=manifest,
        scope='Every panel keeps its evidence tier. Isolated operators and structural work are not measured serving gains. No different starts are merged into one timeline.'))


def timeline(plt,gate,kernels,save):
    """Draw one actual start. Uncaptured GPU time stays unmeasured."""
    name=gate['names'][1]
    folder=ROOT/'raw'/name
    if not (folder/'window.json').exists():return
    window=json.loads((folder/'window.json').read_text());origin=window['start']
    method=gate['method'];cap=gate['cap']
    n_axes=2 if method=='Agg' else 3
    fig,axes=plt.subplots(n_axes,1,figsize=(13,6.5 if n_axes==3 else 4.5),sharex=True,
        gridspec_kw={'height_ratios':[1,1]+([2] if n_axes==3 else [])})
    palette={'GEMM':'#0072b2','Attention':'#d55e00','Norm':'#009e73','Activation':'#cc79a7','Other':'#888888'}
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=color,label=label) for label,color in palette.items()],
        loc='upper right',ncol=5,fontsize=7)
    expected=('replica0','replica1') if method=='Agg' else ('prefill','decode')
    for ax,role in zip(axes,expected):
        data=kernels['roles'].get(role)
        if data is None:
            ax.text(.5,.5,role+' kernels NOT_MEASURED: capture failed',transform=ax.transAxes,ha='center')
            ax.set_yticks([])
            continue
        lo,hi=data['capture_monotonic']
        ax.axvspan(lo-origin,hi-origin,color='#dddddd',alpha=.4,label='Actual capture window')
        # One collection per category keeps all measured intervals. One
        # collection per kernel makes the large diagnostic traces too slow.
        for category,color in palette.items():
            spans=[(event['start']-origin,event['end']-event['start'])
                for event in data['kernel_intervals'] if event['category']==category]
            if spans:
                ax.broken_barh(spans,(.15,.7),facecolors=color,rasterized=True)
        ax.set_yticks([]);ax.set_ylabel(role+' kernels',fontsize=8)
        ax.set_title('Actual CUDA kernels. Outside the shaded capture window: NOT_MEASURED.',fontsize=9)
    if method!='Agg':
        ax=axes[-1]
        source={r['request_id']:r['actual_send_start'] for r in records(folder/'actual_submissions.jsonl')}
        destination={r['request_id']:r['actual_send_start'] for r in records(folder/'D_actual_submissions.jsonl')}
        ready={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
        requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
        for i in range(min(8,len(requests))):
            rid='q'+str(i);record=requests[rid]
            a,b,c,d=source[rid],ready[rid],destination[rid],record['token_events'][0]['worker_time']
            for lo,hi,color in ((a,b,'#a6cee3'),(b,c,'#cccccc'),(c,d,'#cab2d6')):
                if hi>=lo:ax.broken_barh([(lo-origin,hi-lo)],(i-.25,.5),facecolors=color)
            ax.plot(b-origin,i,marker='|',color='#0072b2',ms=8)
            ax.plot(c-origin,i,marker='>',color='#6a3d9a',ms=5)
            ax.plot(record['first_token']-origin,i,marker='*',color='black',ms=6)
        ax.set_yticks(range(8),[f'P_req{i} → D_req{i}' for i in range(8)],fontsize=7)
        ax.invert_yaxis()
        ax.set_title('HOST enclosures: source in-flight (blue), ready-to-admit wait (gray), D preparation (purple). | Cut ready; > D submit; * first output.',fontsize=8)
        ax.set_xlabel('Seconds from the first offered arrival of this same start')
    else:
        axes[-1].set_xlabel('Seconds from the first offered arrival of this same start')
    status='observer gate PASS' if gate['passed'] else 'DIAGNOSTIC: observer gate FAIL'
    fig.suptitle(f'{method} Output{cap}, N36, QPS6 | {status} | Same-run host and GPU clocks')
    fig.text(.01,.01,'One diagnostic start with A/B/A controls. No physical idle claim if the 5% gate fails. PD uses a whole-cohort guard: new suffix work does not preempt an active Decode cohort.',fontsize=7)
    fig.tight_layout(rect=(0,.04,1,.95))
    save(fig,f'root_actual_timeline_{method}_O{cap}'+gate.get('figure_suffix',''),kernels['evidence'],dict(name=name,observer=gate))


if __name__=='__main__':
    main()

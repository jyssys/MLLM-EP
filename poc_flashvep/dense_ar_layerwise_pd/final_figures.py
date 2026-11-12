"""Plot measured intervals without filling unknown time with idle claims."""
import csv,json,statistics
from .common import *
from .analyze import records
from .physical_analysis import merge


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    out=ROOT/'figures';out.mkdir(exist_ok=True);manifest=[]
    def save(fig,name,kind,sources,scope):
        fig.tight_layout();fig.savefig(out/(name+'.svg'));fig.savefig(out/(name+'.png'),dpi=150)
        plt.close(fig);manifest.append(dict(name=name,evidence=kind,sources=sources,scope=scope))
    # Use a qualified Eager observation. Never average rejected traces into it.
    screens=[]
    for p in sorted((ROOT/'analysis').glob('eager_physical_s*_screen_v23.json')):
        r=json.loads(p.read_text())
        if r['trace_qualified'] and r['P_gate']:screens.append((p,r))
    if screens:
        path,r=screens[0];name=r['names'][1];folder=ROOT/'raw'/name
        left=r['P_activity']['window_start'];right=r['P_activity']['window_end'];width=min(3,right-left)
        with (folder/'qualified_scope_epochs.csv').open() as f:epochs=list(csv.DictReader(f))
        with (folder/'kernel_union_prefill.csv').open() as f:P=list(csv.DictReader(f))
        fig,ax=plt.subplots(figsize=(10,3))
        ax.broken_barh([(0,width)],(-.25,.5),facecolors='#eeeeee',hatch='//',edgecolors='#bbbbbb')
        for rows,kind,y,color in [(P,'kernel_union',1,'#e7a146'),(epochs,'D_measured_kernel_inactive',0,'#2caa8c')]:
            spans=[]
            for v in rows:
                if v['kind']!=kind:continue
                a=max(left,float(v['start_monotonic']));b=min(left+width,float(v['end_monotonic']))
                if b>a:spans.append((a-left,b-a))
            ax.broken_barh(spans,(y-.25,.5),facecolors=color)
        ax.set(xlim=(0,width),ylim=(-.6,1.6),yticks=[0,1],yticklabels=['D observed inactivity / UNKNOWN','P kernel union'],xlabel='Seconds from the selected window',title='Native PD: actual CUDA activity and censored D inactivity')
        ax.legend(handles=[Patch(facecolor='#e7a146',label='Kernel union'),Patch(facecolor='#2caa8c',label='Measured kernel inactivity'),Patch(facecolor='#eee',hatch='//',label='Uncaptured D time: UNKNOWN')],loc='upper center',bbox_to_anchor=(.5,-.25),ncol=3)
        save(fig,'figure_1_native_physical_timeline','TRACE_ANALYSIS',[str(path.relative_to(ROOT))],
            'Three-second excerpt from one qualified Start. D inactivity is observed only inside captured empty-engine epochs. Unknown D time is hatched.')
        native=json.loads(next(folder.glob('native_prefill_*.json')).read_text());samples=[v for v in native['queue_samples'] if left<=v['time']<=right]
        fig,ax=plt.subplots(3,1,figsize=(10,5),sharex=True)
        ax[0].plot([v['time']-left for v in samples],[v['waiting'] for v in samples],drawstyle='steps-post',label='Native P waiting queue')
        ax[0].set(ylabel='Requests',title='P compute-ready queue, actual kernel activity, and credit')
        intervals=[(float(v['start_monotonic']),float(v['end_monotonic'])) for v in P if v['kind']=='kernel_union']
        times=[];busy=[];start=left
        while start<right:
            end=min(start+.5,right);times.append((start+end)/2-left)
            busy.append(100*sum(max(0,min(end,b)-max(start,a)) for a,b in intervals)/(end-start));start=end
        ax[1].plot(times,busy,label='Kernel union, 0.5-second bins');ax[1].axhline(80,color='black',linestyle='--')
        ax[1].set(ylabel='Kernel busy %',ylim=(0,105))
        ax[2].plot([v['time']-left for v in samples],[v['KV_free_blocks'] for v in samples],label='P free KV blocks')
        ax[2].axhline(500,linestyle='--',color='black',label='One 8000-token prompt')
        ax[2].set(ylabel='Free blocks',xlabel='Seconds from the common window')
        for a in ax:a.legend(loc='best');a.grid(alpha=.2)
        save(fig,'figure_2_P_queue_and_credit','TRACE_ANALYSIS',[str(path.relative_to(ROOT))],
            'The detailed queue-status audit confirms uncomputed prompt waiters. The title does not imply that Coordinator outstanding requests are compute-ready.')
        durations=[1000*(float(v['end_monotonic'])-float(v['start_monotonic'])) for v in epochs if v['kind']=='D_measured_kernel_inactive']
        fig,ax=plt.subplots(figsize=(8,3));ax.hist(durations,bins=[0,.1,.5,1,2,5,10,20,50,100,250,500,1000]);ax.set(xscale='symlog',xlim=(0,1000),xlabel='Captured kernel-inactive interval (ms)',ylabel='Count',title='Selected D inactivity distribution, not the full-run distribution')
        save(fig,'figure_3_censored_D_idle_windows','TRACE_ANALYSIS',[str(path.relative_to(ROOT))],
            'Only qualified captured intervals appear. These intervals are not certified Decode-safe compute budgets.')
    pilot=ROOT/'analysis/eager_N64_comparison.json'
    if pilot.exists():
        rows=json.loads(pilot.read_text())['rows'];labels=['Native','Guard','Last 1','Last 2','Last 4','Last 8','Colocated']
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,key,title in zip(axes,['TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms'],['TTFT p95','ITL p99','TPOT p95']):
            ax.bar(labels,[v['ratios'][key] for v in rows],color=['#777']*2+['#2caa8c']*4+['#5186b4'])
            ax.axhline(1,color='black',linewidth=.8)
            if key!='TTFT_p95_ms':ax.axhline(1.05,color='#cc3333',linestyle='--')
            ax.set(title=title,ylabel='Ratio to Native');ax.tick_params(axis='x',rotation=50);ax.grid(axis='y',alpha=.2)
        save(fig,'figure_4_static_layer_selection','MEASURED_MARKER_FREE_SELECTION',[str(pilot.relative_to(ROOT))],
            'N64, one Start per arm. The red Decode line is the unchanged 5% target. No confidence interval or sustainable capacity claim.')
    final=ROOT/'analysis/eager_N128_confirmation.json'
    if final.exists():
        data=json.loads(final.read_text());fig,axes=plt.subplots(1,4,figsize=(13,3.5))
        for ax,key in zip(axes,['TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','TPOT_p95_ms']):
            for x,label,color in [(0,'layer','#2caa8c'),(1,'colocated','#5186b4')]:
                z=data['paired_statistics'][label][key];point=z['paired_geometric_mean'];a,b=z['CI95']
                ax.errorbar(x,point,yerr=[[point-a],[b-point]],fmt='o',color=color,capsize=4)
                ax.scatter([x-.07+i*.07 for i in range(len(z['start_ratios']))],z['start_ratios'],s=12,color=color,alpha=.6)
            ax.axhline(1,color='black',linewidth=.8)
            if key.startswith(('ITL','TPOT')):ax.axhline(1.05,color='#cc3333',linestyle='--')
            ax.set(xticks=[0,1],xticklabels=['Layer','Colocated'],ylabel='Ratio to Native',title=key.replace('_ms','').replace('_',' '));ax.grid(axis='y',alpha=.2)
        save(fig,'figure_5_heldout_N128_confirmation','MEASURED_MARKER_FREE_PAIRED_STARTS',[str(final.relative_to(ROOT))],
            'Paired Start-unit bootstrap intervals. Individual Start ratios remain visible. Three starts have limited statistical power.')
    diag=ROOT/'analysis/Decode_event_diagnostic.json'
    if diag.exists():
        data=json.loads(diag.read_text());selected=[]
        for row in data['rows']:
            p=ROOT/'raw'/row['name']/'CUDA_event_enclosures_prefill.json'
            if not p.exists():continue
            for role in ('prefill','decode'):
                values=json.loads((p.parent/f'CUDA_event_enclosures_{role}.json').read_text())['rows']
                values=[v for v in values if 'q4' in v['requests']]
                if values:selected.append((row['name'],role,values))
        if selected:
            fig,ax=plt.subplots(figsize=(11,4.8));labels=[]
            for i,(name,role,values) in enumerate(selected):
                reqs={v['sample_id']:v for v in records(ROOT/'raw'/name/'requests.jsonl')};start=reqs[4]['arrival']
                labels.append(name.split('N32_')[1].split('_event')[0]+' '+role)
                for v in values:
                    color='#e7a146' if role=='prefill' else '#2caa8c' if v['prefilling'] else '#5186b4'
                    ax.broken_barh([(1000*(v['start_monotonic']-start),v['elapsed_ms'])],(i-.3,.6),facecolors=color,hatch='//',edgecolors='white',linewidth=.3)
                ax.scatter([1000*(reqs[4]['first_token']-start)],[i],color='#bb3333',s=15,zorder=5)
            ax.set(yticks=list(range(len(labels))),yticklabels=labels,xlabel='Milliseconds from request q4 arrival',title='Actual stream-event enclosures, not physical kernel intervals')
            ax.grid(axis='x',alpha=.2)
            save(fig,'figure_6_request_execution_enclosures','DIAGNOSTIC_CUDA_EVENT_AND_HOST_TRACE',[str(diag.relative_to(ROOT))],
                'Hatched intervals are CUDA-event enclosures. Red points mark measured first output. Gaps are not labeled GPU idle. Independent Start origins align at the same request arrival.')
    write(out/'figure_manifest.json',dict(figures=manifest,
        missing_data_policy='No unexecuted arm is filled with zero. Structural diagrams and modeled estimates are not mixed into measured timelines.'))
    print('FIGURES',len(manifest))


if __name__=='__main__':main()

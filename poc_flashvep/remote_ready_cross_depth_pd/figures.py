"""Plot observed selections and Host notifications, never synthetic GPU idle."""
import json
from .common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    runs=json.loads((ROOT/'analysis/observed_opportunities.json').read_text())['runs'];manifest=[]
    def save(fig,name,scope):
        fig.tight_layout();fig.savefig(ROOT/'figures'/(name+'.svg'))
        fig.savefig(ROOT/'figures'/(name+'.png'),dpi=160);plt.close(fig)
        manifest.append(dict(name=name,evidence='TRACE_ANALYSIS',scope=scope,
            source='analysis/observed_opportunities.json',new_GPU_execution=False))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,model in zip(axes,('ouro','huginn')):
        selected=[r for r in runs if r['model']==model]
        a=[r['observed_any_aligned_pair_batches'] for r in selected]
        b=[r['observed_cross_depth_only_batches'] for r in selected]
        ax.bar([0,1],a,label='Contains an aligned Prefill/Decode pair',color='#e7a146')
        ax.bar([0,1],b,bottom=a,label='No aligned Prefill/Decode pair',color='#2caa8c')
        for i,r in enumerate(selected):
            ax.annotate(f"{a[i]+b[i]} used / {r['Decode_core_executions']} Decode core calls",
                (i,a[i]+b[i]),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        ax.set(title=model.capitalize()+' — existing Native Mixed',xticks=[0,1],
            xticklabels=['L128, N8, cap32','L256, N8, cap16'],ylabel='Observed remote mixed calls',ylim=(0,max(a[i]+b[i] for i in (0,1))*1.3))
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(loc='upper left',fontsize=7)
    save(fig,'figure_1_already_used_remote_opportunities',
        'Four historical N8 diagnostic cells. Input lengths are not request counts. An aligned pair does not make every row share one depth. Unused legal opportunities remain NOT_MEASURED.')

    frames=json.loads((ROOT/'raw/selected_D_core_frames.json').read_text())
    requests=json.loads((ROOT/'raw/remote_request_chronology.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(12,6))
    for ax,model in zip(axes,('ouro','huginn')):
        name=f'diag256_v2_{model}_A_native_mixed_0'
        first=next(b for b in frames if b['run']==name and b['remote_prefill_rows'])
        tid=next(s['request_id'] for s in first['spans'] if s['phase']=='prefill')
        request=next(r for r in requests if r['run']==name and r['transfer_id']==tid)
        origin=request['source_cut_snapshot_Host'];end=first['host_end']+.01
        receive=request['D_receive_Host_observation'];active=request['D_activation_Host_observation']
        if receive>origin:ax.broken_barh([(0,1000*(receive-origin))],(.8,.35),facecolors='#aaa',hatch='//')
        ax.scatter([0],[0],color='#c44',marker='|',s=180)
        ax.scatter([1000*(receive-origin)],[1],color='#555',s=25)
        ax.scatter([1000*(active-origin)],[2],color='#2caa8c',s=25)
        for b in frames:
            if b['run']!=name or b['host_end']<origin or b['host_begin']>end:continue
            left=max(origin,b['host_begin']);right=min(end,b['host_end'])
            if right>left:
                color='#2caa8c' if b['remote_prefill_rows'] else '#5186b4'
                ax.broken_barh([(1000*(left-origin),1000*(right-left))],(2.8,.35),facecolors=color,hatch='//')
        decode=sorted({s['depth'] for s in first['spans'] if s['phase']=='decode'})
        prefill=sorted({s['depth'] for s in first['spans'] if s['phase']=='prefill'})
        ax.set(yticks=[0,1,2,3],yticklabels=['P snapshot Host notice','Import Host observation','Activation Host observation','D model Host enclosures'],
            xlabel='Milliseconds from the selected Source snapshot notice',title=f'{model.capitalize()} {request["request_id"]}: frame {first["batch_id"]}, Decode depths {decode}, Prefill depths {prefill}')
        ax.grid(axis='x',alpha=.2)
    axes[0].legend(handles=[Patch(facecolor='#5186b4',hatch='//',label='Decode Host enclosure'),
        Patch(facecolor='#2caa8c',hatch='//',label='Mixed Host enclosure')],loc='upper left',fontsize=8)
    save(fig,'figure_2_actual_remote_import_to_mixed_host_observations',
        'Host observations from old diagnostics. Hatching marks enclosures, not GPU kernels. Snapshot time does not time GPU readiness. Blank regions do not imply physical idle. Depth labels are zero-based.')
    write(ROOT/'figures/figure_manifest.json',dict(figures=manifest))
    print('FIGURES',len(manifest))


if __name__=='__main__':main()

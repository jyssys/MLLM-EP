"""Plot measured data and explicit missing evidence. Never draw invented idle."""
import collections
import csv
import statistics
from .common import *
from .analyze import records,table
from .profile_analysis import union


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'svg.fonttype':'none'})
    destination=ROOT/'figures';destination.mkdir(exist_ok=True)
    runs=json.loads((ROOT/'serving_runs.json').read_text())
    configs={p.stem:json.loads(p.read_text()) for p in (ROOT/'configs').glob('*.json')}
    manifest=[];extracts=[]
    def save(fig,name,sources,scope):
        fig.tight_layout()
        outputs=[]
        for suffix in ('png','svg'):
            path=destination/(name+'.'+suffix)
            fig.savefig(path,dpi=180,bbox_inches='tight');outputs.append(str(path.relative_to(ROOT)))
        plt.close(fig)
        manifest.append(dict(name=name,outputs=outputs,exact_sources=sources,scope=scope,
            source_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/s for s in sources) if p.is_file()}))
    def pick(model):
        profiled=[]
        for p in (ROOT/'raw').glob(f'long_profile_{model}_*/profile_analysis.json'):
            proof=json.loads(p.read_text())
            if proof.get('eligible_for_matched_clean'):profiled.append(p.parent.name)
        if profiled:return sorted(profiled)[0]
        selected=ROOT/'state/confirmation_plan.json'
        if selected.exists():
            p=json.loads(selected.read_text())
            names=[r['PD'] for r in p['runs'] if r['model']==model]
            for name in names:
                if (ROOT/'raw'/name/'window.json').exists():return name
        values=[r for r in runs if r['model']==model and r['phase']=='qps_pilot' and r['status']=='COMPLETE' and
            r['length']==(6144 if model=='ouro' else 2560)]
        return max(values,key=lambda r:r.get('P_completion_rate_central') or 0)['name'] if values else None
    for model,length in [('ouro',6144),('huginn',2560)]:
        new_epoch=any(configs.get(r['name'],{}).get('measurement_epoch')=='WARM_GRAPH_DURABLE_V2' and r['status']=='COMPLETE' for r in runs if r['model']==model)
        groups=collections.defaultdict(list)
        sources=['serving_runs.json']
        for row in runs:
            cfg=configs.get(row['name'],{})
            if row['model']!=model or row['length']!=length or row['status']!='COMPLETE':continue
            if new_epoch and cfg.get('measurement_epoch')!='WARM_GRAPH_DURABLE_V2':continue
            if cfg.get('nsys') or cfg.get('state_probe') or cfg.get('verify_transfer'):continue
            if cfg.get('phase') not in ('qps_pilot','aggregated_QPS_pilot'):continue
            if cfg['name'].startswith('generic_control_'):
                if cfg['name'].split('_q')[0] not in (f'generic_control_{model}_credit8g_packed',f'generic_control_{model}_Pslots2_packed'):continue
                label='PD supported tuning + exact CPU packing'
            elif cfg['arm'].startswith('DP'):label='DP2 Native Mixed C'+str(cfg['chunk'])
            else:
                label='PD Atomic' if cfg['atomic_prompt'] else 'PD C'+str(cfg['chunk'])
                label+=' | slots P'+str(cfg['P_slots'])+'/D'+str(cfg['D_slots'])
            groups[label].append(row)
            extracts.append(dict(figure=model+'_duetserve_style',series=label,**{k:row.get(k) for k in
                ('name','target_QPS','TTFT_ms_p95','request_TPOT_ms_p95','token_ITL_ms_p99','total_tokens_per_s','output_tokens_per_s')}))
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for label,values in sorted(groups.items()):
            values.sort(key=lambda r:r['target_QPS'])
            for axis,field in zip(axes,('TTFT_ms_p95','request_TPOT_ms_p95','total_tokens_per_s')):
                axis.plot([r['target_QPS'] for r in values],[r[field] for r in values],'o-',label=label)
        for axis,title,y in zip(axes,('Request latency','Decode latency','Total token throughput'),('TTFT p95 (ms)','Request TPOT p95 (ms)','Input + output tokens/s')):
            axis.set(xlabel='Target offered requests/s',ylabel=y,title=title);axis.set_ylim(bottom=0)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'{model.title()} | {length}-token input | N=24 exploratory starts\nControlled ignore-EOS. Finite arrival-to-drain measurements. No independent-start CI for this sweep.',fontsize=11,y=1.06)
        save(fig,model+'_duetserve_style',sources,'Each curve retains one native execution/transport setting. No cross-setting envelope or sustainable-capacity claim.')
        fig,ax=plt.subplots(figsize=(7,4))
        for label,values in sorted(groups.items()):
            values.sort(key=lambda r:r['target_QPS'])
            ax.plot([r['target_QPS'] for r in values],[r['token_ITL_ms_p99'] for r in values],'o-',label=label)
        slo=json.loads((ROOT/'configs/frozen_SLO.json').read_text()).get(f'{model}_L{length}')
        if slo:ax.axhline(slo['ITL_ms'],color='black',ls='--',label='Frozen token ITL limit')
        ax.set(xlabel='Target offered requests/s',ylabel='Token ITL p99 (ms)',title=f'{model.title()} | Token tail is not request-average TPOT')
        ax.set_ylim(bottom=0);ax.legend(fontsize=7)
        save(fig,model+'_token_itl_tail',['serving_runs.json','configs/frozen_SLO.json'],'Observed coordinator token-delivery intervals. Bundled deliveries retain the documented interpolation limit.')
        name=pick(model)
        if name is None:continue
        cfg=configs[name];folder=ROOT/'raw'/name
        slo=json.loads((ROOT/'configs/frozen_SLO.json').read_text()).get(f'{model}_L{cfg["length"]}')
        window=json.loads((folder/'window.json').read_text());start=window['start']
        from .queues import analyze as queue_analysis
        q,series=queue_analysis(name,window)
        fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
        axes[0].plot([r['time']-start for r in series],[r['ready_backlog_lower_bound'] for r in series],label='D-clear P-ready lower bound')
        axes[0].plot([r['time']-start for r in series],[r['coordinator_D_credit_blocked'] for r in series],label='D-credit-blocked requests')
        axes[0].set_ylabel('Requests');axes[0].legend(fontsize=8)
        profile=folder/'profile_analysis.json'
        kernels=folder/'aligned_kernels.json'
        if profile.exists() and kernels.exists():
            proof=json.loads(profile.read_text());ks=json.loads(kernels.read_text())['kernels']
            from .interval_bins import bins
            intervals=bins(ks,start,window['end'],.5)
            edges=[a for a,b,v in intervals];busy=[v for a,b,v in intervals]
            axes[1].step([t-start for t in edges],busy,where='post')
            axes[1].set_ylabel('P kernel union (%)');axes[1].set_ylim(0,105)
            axes[1].set_title('MEASURED device intervals'+(' | qualified profile' if proof['eligible_for_matched_clean'] else ' | DIAGNOSTIC: perturbation gate failed'),fontsize=9)
        else:
            axes[1].text(.5,.5,'P GPU kernel activity: NOT_MEASURED in this start',ha='center',transform=axes[1].transAxes)
            axes[1].set_yticks([])
        for path in folder.glob('memory_progress_*.jsonl'):
            values=[r for r in records(path) if r['role']=='decode' and start<=r['time']<=window['end']]
            if values:axes[2].step([r['time']-start for r in values],[r['KV_free_blocks']*r['KV_block_bytes']/1024**3 for r in values],where='post',label='D owner free KV')
        axes[2].set(xlabel='Seconds after measured arrival start',ylabel='Free KV (GiB)');axes[2].legend(fontsize=8)
        for ax in axes:ax.axvline(window['last_arrival']-start,color='grey',ls='--');ax.set_xlim(0,window['end']-start)
        fig.suptitle(f'{model.title()} | {name}\nQueue and credit evidence are not a GPU compute-saturation proof.',fontsize=10)
        save(fig,model+'_p_queue_and_d_credit',[f'raw/{name}/window.json',f'raw/{name}/admission_reasons.jsonl']+
            [str(p.relative_to(ROOT)) for p in folder.glob('memory_progress_*.jsonl')],
            'All panels share one start and host monotonic clock. Missing kernel activity is not drawn as zero or idle.')
        # Unique-storage inventory and live CUDA memory are separate from KV
        # occupancy. Do not add the active KV count to its containing pool.
        inventories=[]
        for path in folder.glob('memory_inventory_initial_*.json'):
            v=json.loads(path.read_text())
            pid=path.stem.rsplit('_',1)[1]
            snapshots=records(folder/f'memory_progress_{pid}.jsonl')
            if not snapshots:continue
            s=min(snapshots,key=lambda r:abs(r['time']-v['time']))
            inventories.append((v,s,path))
        if inventories:
            fig,ax=plt.subplots(figsize=(10,4));labels=[]
            fields=[('weight_and_model_buffer_storage_bytes','Weights + buffers'),('KV_storage_bytes','KV pool (reserved)'),
                ('exposed_runner_workspace_storage_bytes','Exposed workspace'),('exposed_NIXL_hidden_staging_storage_bytes','NIXL staging'),
                ('exposed_graph_tensor_storage_bytes','Exposed graph tensors'),('opaque_or_unattributed_allocator_bytes','Opaque allocator')]
            bottom=[0.]*len(inventories)
            for field,label in fields:
                values=[v[field]/1024**3 for v,s,p in inventories]
                ax.bar(range(len(values)),values,bottom=bottom,label=label)
                bottom=[a+b for a,b in zip(bottom,values)]
            total=statistics.mean(s['device_total_bytes'] for v,s,p in inventories)/1024**3
            ax.axhline(total,color='black',ls='--',label=f'CUDA-visible capacity {total:.3f} GiB')
            ax.axhline(79.6474609375,color='grey',ls=':',label='NVML total HBM 79.647 GiB')
            for i,(v,s,p) in enumerate(inventories):
                ax.text(i,bottom[i]+.2,f'{bottom[i]:.2f} GiB\n{100*bottom[i]/79.6474609375:.1f}% NVML HBM',ha='center',fontsize=9)
            ax.set_xticks(range(len(inventories)),[v['role'].title() for v,s,p in inventories])
            ax.set(ylabel='Binary GiB',title=f'{model.title()} | Actual owner allocation, {name}')
            ax.set_ylim(0,total*1.18);ax.legend(fontsize=7,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.43))
            save(fig,model+'_hbm_budget',[str(p.relative_to(ROOT)) for v,s,p in inventories],
                'Unique CUDA storage categories. The opaque residual includes unexposed graph/library allocation. Initial per-owner snapshots are not simultaneous cross-worker peaks. Active KV is contained in the pool.')
        fig,ax=plt.subplots(figsize=(8,4))
        row=next(r for r in runs if r['name']==name)
        ax.scatter([0],[row['request_TPOT_ms_p95']],label='Measured no-borrow request TPOT p95')
        ax.scatter([0],[row['token_ITL_ms_p99']],marker='x',label='Measured no-borrow token ITL p99')
        if slo:
            ax.axhline(slo['TPOT_ms'],ls='--',color='C0',label='Frozen TPOT limit')
            ax.axhline(slo['ITL_ms'],ls='--',color='C1',label='Frozen ITL limit')
        ax.text(.55,.65,'Positive borrowed-work points:\nNOT_CONFIRMED',transform=ax.transAxes,ha='center')
        ax.set(xlabel='Additional real D-side Prefill tokens',ylabel='Decode latency (ms)',title=f'{model.title()} | No borrowing claim without an eligible measured probe')
        ax.set_xlim(-5,128);ax.set_ylim(bottom=0);ax.legend(fontsize=8)
        save(fig,model+'_safe_borrowing_status',[f'raw/{name}/requests.jsonl','configs/frozen_SLO.json'],
            'Only zero added-work baseline points are measured here. No synthetic GEMM or untested positive point is shown.')
    # A compact evidence map retains missing observations explicitly.
    verdict=ROOT/'final_per_model_verdict.json'
    fig,ax=plt.subplots(figsize=(9,5));ax.axis('off')
    rows=['P-ready compute backlog','P kernel saturation','D KV credit','Decode TPOT + ITL margin','Actual safe D borrowing']
    values=json.loads(verdict.read_text()) if verdict.exists() else {}
    cells=[]
    for field in ('P_ready','P_compute','D_credit','D_SLO','D_borrowing'):
        cells.append([values.get(m,{}).get(field,'NOT_CONFIRMED') for m in ('ouro','huginn')])
    display=ax.table(cellText=cells,rowLabels=rows,colLabels=['Ouro R4','Huginn R32'],loc='center',cellLoc='center',colWidths=[.3,.3])
    display.auto_set_font_size(False);display.set_fontsize(10);display.scale(1,2.0)
    ax.set_title('Same-start Prefill-heavy proof | Missing evidence is not a pass',pad=20)
    save(fig,'executive_gate_map',['final_per_model_verdict.json'] if verdict.exists() else [],
        'A strict target requires all causal gates and independent repetitions. This table does not infer recurrence impossibility.')
    table('figure_data_extract.csv',extracts)
    write(ROOT/'figure_manifest.json',dict(created=time.time(),figures=manifest,
        plotting_source=str(Path(__file__).relative_to(REPO)),plotting_source_SHA256=digest(Path(__file__))))
    print(json.dumps(dict(figures=len(manifest))))


if __name__=='__main__':main()

"""Plot only measured starts and label host, model and missing evidence."""
import json
import shutil
from .common import ROOT,REPO,write


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42})
    source=ROOT/'analysis/heldout_start_statistics.json'
    data=json.loads(source.read_text());out=ROOT/'figures';out.mkdir(exist_ok=True)
    colors=dict(Native='#555555',Colocated='#d18b15',Token='#377eb8',Static='#4daf4a',Dynamic='#e41a1c')
    labels=dict(Native='Native PD',Colocated='Colocated B8192/S1024',
        Token='Tuned Token',Static='Tuned Static Layer',Dynamic='Online Dynamic Layer')
    figures=[]
    def save(fig,stem,evidence,scope):
        fig.tight_layout(rect=(0,.12,1,.93) if stem=='13_actual_dynamic_cut_distribution' else None)
        for suffix in ('.png','.pdf'):
            fig.savefig(out/(stem+suffix),dpi=180,bbox_inches='tight')
        plt.close(fig);figures.append(dict(stem=stem,evidence=evidence,scope=scope))
    curves=(('01_mean_TTFT','mean_TTFT_ms','Mean TTFT (s)',.001),
        ('02_mean_TBT','mean_TBT_ms','Mean TBT (ms)',1.),
        ('03_ITL_p99','ITL_p99_ms','ITL p99 (ms)',1.),
        ('04_raw_output_tokens_s','output_tokens_s','Generated output tokens/s',1.),
        ('05_joint_SLO','Joint_attainment','Joint SLO attainment (%)',100.),
        ('06_SLO_output_tokens_s','Output_token_SLO_goodput','SLO-qualified output tokens/s',1.))
    for stem,key,label,scale in curves:
        fig,axes=plt.subplots(1,2,figsize=(10,3.4))
        for ax,cap in zip(axes,(16,64)):
            for method,color in colors.items():
                points=sorted((r for r in data['aggregate'] if r['cap']==cap and r['method']==method),key=lambda r:r['QPS'])
                if not points:
                    continue
                x=[r['QPS'] for r in points];y=[r['metrics'][key]['mean']*scale for r in points]
                ax.plot(x,y,'o-',label=labels[method],color=color,lw=1.4,ms=4)
                for a,b,row in zip(x,y,points):
                    ci=row['metrics'][key]['CI95']
                    if ci:
                        ax.errorbar(a,b,yerr=[[max(0.,b-ci[0]*scale)],[max(0.,ci[1]*scale-b)]],color=color,capsize=2)
                    ax.annotate(str(row['starts']),(a,b),xytext=(3,4),textcoords='offset points',fontsize=6,color=color)
            ax.set(title=f'Input8000 / Output{cap}',xlabel='Offered QPS (requests/s)',ylabel=label,xlim=(1.7,10.3))
            ax.set_xticks([2,4.8,6.4,10]);ax.legend(fontsize=7)
            if key=='Joint_attainment':
                ax.axhline(95,color='black',ls=':',lw=.7);ax.set_ylim(0,103)
            if key=='ITL_p99_ms':
                ax.set_yscale('log');ax.axhline(30 if cap==16 else 31,color='black',ls=':',lw=.7)
        regime='; frozen Decode-priority SLO' if key in ('Joint_attainment','Output_token_SLO_goodput') else ''
        fig.suptitle('Unseen finite cohorts; labels = starts; no CI for n=1'+regime,fontsize=9)
        save(fig,stem,'MEASURED','Start-mean curves. Start bootstrap CIs. Frozen Decode-priority SLO. Missing QPS are not measured. No sustainable-capacity claim.')
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for ax,cap in zip(axes,(16,64)):
        for method,color in colors.items():
            points=[r for r in data['aggregate'] if r['cap']==cap and r['method']==method]
            for row in points:
                ax.scatter(row['metrics']['TTFT_p95_ms']['mean']/1000,row['metrics']['ITL_p99_ms']['mean'],
                    s=35,c=color,label=labels[method] if row==points[0] else None)
                ax.annotate(f"q{row['QPS']:g}, n={row['starts']}",
                    (row['metrics']['TTFT_p95_ms']['mean']/1000,row['metrics']['ITL_p99_ms']['mean']),fontsize=6)
        ax.set(title=f'Output{cap}',xlabel='TTFT p95 (s)',ylabel='ITL p99 (ms)');ax.legend(fontsize=7)
    save(fig,'07_axis_Pareto','MEASURED','Per-start quantile means. Same frozen model, pair and workload within each QPS. Points at different QPS are not a matched-throughput proof.')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));model_path=ROOT/'analysis/strong_per_cap_fraction_oracle_v2_Joint95.json'
    if model_path.exists():
        model=json.loads(model_path.read_text());values=[model['static']['best_with_Joint95']['metric']['SLO_output_tokens_s'],model['best_tested_dynamic_with_Joint95']['metric']['SLO_output_tokens_s']]
        axes[0].bar(['Fixed cap/fraction','Bounded dynamic beam'],values,color=['#4daf4a','#999999'])
        axes[0].set(ylabel='Modeled SLO output tokens/s',title='OFFLINE_ORACLE; Joint95 constraint; no bound')
    else:
        axes[0].text(.5,.5,'NOT_RUN',ha='center')
    actual_path=ROOT/'analysis/flow_low_headroom_actual_pilot.json'
    if actual_path.exists():
        values=json.loads(actual_path.read_text());labels=[r['name'].replace('flow_low_','').replace('_N128_v1','') for r in values]
        axes[1].bar(labels,[r['SLO_regimes']['Decode_priority']['Output_token_SLO_goodput'] for r in values],color='#377eb8')
        axes[1].tick_params(axis='x',labelrotation=30,labelsize=6);axes[1].set(ylabel='Measured SLO output tokens/s',title='MEASURED pilot; n=1; finite cohort')
    else:
        axes[1].text(.5,.5,'Online pilot NOT_RUN',ha='center')
    save(fig,'08_static_oracle_online','OFFLINE_ORACLE_AND_MEASURED_SEPARATE_PANELS','The left panel is modeled. The right panel is an actual pilot when available. The beam can fall below strong static because it is approximate.')
    fig,ax=plt.subplots(figsize=(8,3.6));found=False
    from .analysis import records
    for name in ('flow4_H3_N128_O16_q6.4_v1','flow8_H2_N64_O64_q4.8_v1','flow_low_Online_H2_N128_v1'):
        path=ROOT/'raw'/name/'online_decisions.jsonl';samples=[]
        for row in records(path):
            deadlines=row['observation'].get('own_TTFT_deadlines',[])
            cut=row['choice']['layers']
            candidate=next((v for v in row['choice']['candidates'] if v['layers']==cut),None)
            if deadlines and candidate:
                samples.append(1000*(min(deadlines)-candidate['predicted_D_finish']))
        if samples:
            ax.hist(samples,bins=15,alpha=.4,label=name.replace('_v1',''));found=True
    if found:
        ax.legend(fontsize=6)
    else:
        ax.text(.5,.5,'Budget trace NOT_RUN',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Predicted first-output deadline residual (ms)',ylabel='Source decisions',
        title='Host-state prediction; not physical GPU idle or measured safe kernel slack')
    save(fig,'09_host_budget_variance','TRACE_ANALYSIS','Use actual decision observations and candidate predictions. No physical GPU-idle inference.')
    fig,axes=plt.subplots(1,2,figsize=(8,3.3))
    axes[0].bar(['Max absolute','RMS'],[.015625,.00020561653946060687],color=['#e41a1c','#377eb8'])
    axes[0].set_yscale('log');axes[0].set(title='First captured difference: layer0 o_proj',ylabel='Tensor error; last8 prefill rows')
    axes[1].bar([-.18,.82],[28.125,28.0],width=.35,label='TP1',color='#377eb8')
    axes[1].bar([.18,1.18],[27.875,27.875],width=.35,label='TP2',color='#e41a1c')
    axes[1].set_xticks([0,1],['Token1986','Token785']);axes[1].set_ylim(27.7,28.2)
    axes[1].set(title='Step6, common input prefix',ylabel='Actual captured logit');axes[1].legend()
    save(fig,'10_TP2_first_divergence','CORRECTNESS_ONLY','Strict TP1 parity fails. Two BF16 local projections and their sum reproduce actual TP2. Sampled tensor rows are not a whole-TP2-state qualification.')
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    cut_colors={0:'#aaaaaa',2:'#80b1d3',4:'#8dd3c7',8:'#fdb462',12:'#fb8072'}
    for ax,cap in zip(axes,(16,64)):
        values=[r for r in data['rows'] if r['cap']==cap and r['method']=='Dynamic' and
                r['QPS']==(6.4 if cap==16 else 4.8)]
        values.sort(key=lambda row:row['seed']);bottom=[0.]*len(values)
        for cut,color in cut_colors.items():
            fraction=[100*r['actual_cuts'].get(str(cut),0)/r['count'] for r in values]
            ax.bar(range(len(values)),fraction,bottom=bottom,color=color,label=f'Last{cut}')
            bottom=[a+b for a,b in zip(bottom,fraction)]
        ax.set_xticks(range(len(values)),[str(r['seed'])[-4:] for r in values])
        ax.set(title=f"Output{cap}; QPS{6.4 if cap==16 else 4.8:g}",xlabel='Independent arrival-seed suffix',
            ylabel='Actual request cut fraction (%)',ylim=(0,105))
        if not values:
            ax.text(.5,.5,'NOT_RUN',ha='center',transform=ax.transAxes)
    handles,cut_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,cut_labels,loc='lower center',bbox_to_anchor=(.5,0),ncol=5,fontsize=8)
    fig.suptitle('MEASURED online cut counts; not forced-cut state tests',fontsize=9)
    save(fig,'13_actual_dynamic_cut_distribution','MEASURED','Use completed actual cut commits from the frozen held-out dynamic runs. A diverse histogram alone does not establish benefit over tuned static or matched geometry controls.')
    write(ROOT/'analysis/figure_manifest.json',dict(figures=figures,
        source='analysis/heldout_start_statistics.json',physical_kernel_timeline='NOT_MEASURED'))
    for source_stem,target_stem in (('01_mean_TTFT','mean_ttft'),('02_mean_TBT','mean_tbt'),
        ('04_raw_output_tokens_s','output_tokens_s'),('05_joint_SLO','slo')):
        for suffix in ('.png','.pdf'):
            shutil.copyfile(out/(source_stem+suffix),REPO/'poc_flashvep'/('dense_ar_layer_token_dynamic_'+target_stem+suffix))
    print('FIGURES',len(figures),flush=True)


if __name__=='__main__':
    main()

"""Plot measured start-level curves and label all diagnostic evidence."""
import statistics
from .common import *
from .analyze import records

COLORS=dict(Native='#4275b6',Colocated='#d69235',Static='#b8597c',Revised='#188d89')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    cells=json.loads((ROOT/'analysis/four_method_qps_results.json').read_text())['cells']
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    def curve(key,label,name,factor=1.,regime=None):
        fig,axes=plt.subplots(1,3,figsize=(16,4.5),squeeze=False)
        for ax,cap in zip(axes[0],(16,64,200)):
            for arm,color in COLORS.items():
                series=[]
                for c in cells:
                    if c['cap']!=cap:continue
                    v=c['arms'][arm]
                    e=v['SLO'][regime][key] if regime else v['estimates'][key]
                    if e['mean'] is None:continue
                    series.append((c['rate'],e))
                series.sort(key=lambda v:v[0])
                if not series:continue
                xs=[x for x,_ in series];ys=[e['mean']*factor for _,e in series]
                low=[(e['mean']-e['CI95'][0])*factor if e['CI95'] else 0 for _,e in series]
                high=[(e['CI95'][1]-e['mean'])*factor if e['CI95'] else 0 for _,e in series]
                ax.errorbar(xs,ys,yerr=[low,high],marker='o',capsize=3,label=arm,color=color)
            ax.set(title=f'Input8000 / Output{cap}',xlabel='Offered QPS (requests/s)',ylabel=label)
        axes[0,0].legend()
        fig.suptitle('Independent N128 starts: mean and 95% start-bootstrap interval\nNo curve connects different output lengths. Missing runs are not zero.')
        fig.tight_layout();save(fig,name)
    curve('TTFT_p95_ms','TTFT p95 (s)','figure_A_qps_vs_ttft',.001)
    curve('TPOT_p95_ms','Request TPOT p95 (ms)','figure_B_qps_vs_tpot')
    curve('output_tokens_s','Output tokens/s, including cohort drain','figure_C_qps_vs_generation_throughput')
    curve('ITL_p99_ms','ITL p99 (ms), including first interval','figure_D_qps_vs_itl')
    curve('Joint_attainment','Joint SLO attainment','figure_E_balanced_joint_attainment',regime='Balanced')
    curve('Request_SLO_goodput','Qualified requests/s, including drain','figure_F_balanced_slo_goodput',regime='Balanced')
    curve('Request_SLO_goodput','Qualified requests/s, including drain','figure_K_decode_priority_slo_goodput',regime='Decode_priority')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,cap in zip(axes,(16,64,200)):
        selected=sorted([c for c in cells if c['cap']==cap and c['arms']['Revised']['total_requests']>0],key=lambda c:c['rate'])
        bottom=[0.]*len(selected)
        for layers,color in zip((0,2,4,8,12),('#b8bdc5','#a8d9c3','#73bea4','#379d86','#0a6f65')):
            ys=[]
            for c in selected:
                v=c['arms']['Revised'];ys.append(100*v['layer_distribution'][str(layers)]/v['total_requests'] if v['total_requests'] else 0.)
            ax.bar([str(c['rate']) for c in selected],ys,bottom=bottom,color=color,label='Native' if layers==0 else f'Last{layers}')
            bottom=[a+b for a,b in zip(bottom,ys)]
        ax.set(title=f'Output{cap}',xlabel='Offered QPS',ylabel='Completed request fraction (%)',ylim=(0,100))
    axes[0].legend();fig.suptitle('Actual completed cuts selected from observed online state. Forced qualification is excluded.')
    fig.tight_layout();save(fig,'figure_G_actual_dynamic_cut_distribution')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for c in cells:
        if c['cap']!=16:continue
        for arm,v in c['arms'].items():
            e=v['estimates']
            if e['TTFT_p95_ms']['mean'] is None:continue
            for ax,key in zip(axes,('ITL_p99_ms','TPOT_p95_ms')):
                ax.scatter(e['TTFT_p95_ms']['mean']/1000,e[key]['mean'],color=COLORS[arm],s=45)
                ax.annotate(str(c['rate']), (e['TTFT_p95_ms']['mean']/1000,e[key]['mean']),fontsize=7)
    for ax,key in zip(axes,('ITL p99 (ms)','TPOT p95 (ms)')):ax.set(xlabel='TTFT p95 (s)',ylabel=key)
    axes[0].legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=color,label=arm) for arm,color in COLORS.items()])
    fig.suptitle('Input8000/Output16 start-mean Pareto observations. Labels show offered QPS.')
    fig.tight_layout();save(fig,'figure_H_ttft_decode_pareto')
    write(folder/'manifest.json',dict(files=[dict(path=str(p.relative_to(REPO)),SHA256=digest(p))
        for p in sorted(folder.glob('*')) if p.suffix in ('.png','.svg','.html')],
        scope='Measured independent-start summaries. Throughput and goodput include finite-cohort drain. No inferred physical idle is plotted.'))
    print('PRIMARY_FIGURES_READY')


if __name__=='__main__':main()

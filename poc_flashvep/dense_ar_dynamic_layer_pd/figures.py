"""Draw measured results without substituting unknown GPU idle capacity."""
import math,statistics
from .common import *
from .analyze import records,metrics
from .enclosure_analysis import execution_class


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=170,bbox_inches='tight')
        fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    all_cells=json.loads((ROOT/'analysis/all_cell_results.json').read_text())['results']
    selection=json.loads((ROOT/'configs/heldout_selection.json').read_text())
    held=json.loads((ROOT/'analysis/heldout_results.json').read_text())['results']
    extra=ROOT/'analysis/exploratory_fraction50_results.json'
    if extra.exists():held+=json.loads(extra.read_text())['results']
    stats=json.loads((ROOT/'analysis/heldout_paired_statistics.json').read_text())['results']
    crossover=json.loads((ROOT/'analysis/crossover_discovery_accounting.json').read_text())['results']
    labels=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s','Joint_attainment')
    fig,axes=plt.subplots(5,5,figsize=(21,15),squeeze=False)
    for i,cap in enumerate((16,32,64,128,200)):
        for j,key in enumerate(labels):
            ax=axes[i,j]
            for arm,color in (('Native','#3b6fb6'),('Colocated','#d5852b')):
                values=sorted([(r['offered_QPS'],r[arm+'_metrics']) for r in crossover
                    if r['cap']==cap and r['status']=='MEASURED'],key=lambda x:x[0])
                ys=[m['SLO_regimes']['Balanced'][key] if key=='Joint_attainment' else m[key] for _,m in values]
                ax.plot([q for q,_ in values],ys,'o-',label=arm,color=color)
            if i==0:ax.set_title(key.replace('_',' '))
            if j==0:ax.set_ylabel(f'Output{cap}')
            if i==4:ax.set_xlabel('Offered QPS')
            if key=='Joint_attainment':ax.set_ylim(-.03,1.03)
    axes[0,0].legend()
    fig.suptitle('Figure1 — PD vs qualified Colocated discovery anchors\nN64, one arrival seed per cell. Later strongest selections appear in the held-out figures. No causal crossover claim.',y=1.01)
    fig.tight_layout();save(fig,'figure_1_pd_colocated_crossover')

    fig,axes=plt.subplots(5,1,figsize=(16,15))
    controls={r['arm']:r['interference_gate_passed'] for r in
        json.loads((ROOT/'analysis/enclosure_control_results.json').read_text())['results']}
    diag=json.loads((ROOT/'analysis/diagnostic_decode_intervals_v1.json').read_text())
    left=diag['left'];right=diag['right']
    for y,role in enumerate(('prefill','decode')):
        data=json.loads((ROOT/'analysis'/f'diagnostic_{role}_intervals_v1.json').read_text())
        axes[0].broken_barh([(a-left,b-a) for a,b in data['union']],(y-.25,.5),facecolors=('#e0a14c' if y==0 else '#3b6fb6'))
    axes[0].set(yticks=[0,1],yticklabels=['Native P kernels','Native D kernels'],xlim=(0,right-left),
        title='Input8000/Output32 diagnostic CUPTI kernel union. Interference gate FAILED. Blank time is not clean safe slack.')
    for ax,arm in zip(axes[1:],('Native','Colocated','Static','COST')):
        name=f'enclosure_{arm}_O16_N64_events_v1';raw=ROOT/'raw'/name
        window=raw/'window.json'
        if not window.exists():
            ax.text(.5,.5,'NOT_RUN: CUDA event control',ha='center',transform=ax.transAxes);continue
        start=json.loads(window.read_text())['start']
        roles=(('replica0',0),('replica1',1)) if arm=='Colocated' else (('prefill',0),('decode',1))
        for role,y in roles:
            path=raw/f'CUDA_event_enclosures_{role}.json'
            if not path.exists():continue
            for row in json.loads(path.read_text())['rows']:
                if not row['requests'] or not all(r.startswith('q') for r in row['requests']):continue
                kind=execution_class(row)[0]
                color='#138e88' if kind=='D_SUFFIX' else '#e0a14c' if kind in ('P_PREFIX','COLOCATED_PREFILL') else '#3b6fb6'
                ax.broken_barh([(row['start_monotonic']-start,row['elapsed_ms']/1000)],(y-.3,.6),facecolors=color)
        ax.set(yticks=[0,1],yticklabels=[arm+' '+role+' enclosure' for role,_ in roles],
            title='Input8000/Output16 stream enclosures. Interference gate '+('PASS' if controls[arm] else 'FAILED — DIAGNOSTIC ONLY')+'. Not kernel union or physical idle.')
    for ax in axes:ax.invert_yaxis()
    axes[-1].set_xlabel('Seconds from the corresponding observation start')
    fig.legend(handles=[Line2D([0],[0],color='#e0a14c',lw=5,label='Prefill / P prefix'),
        Line2D([0],[0],color='#3b6fb6',lw=5,label='Full Decode'),
        Line2D([0],[0],color='#138e88',lw=5,label='D suffix continuation')],loc='lower center',ncol=3)
    fig.suptitle('Figure2 — Distinct timing evidence. Each panel uses its own recorded start. No cross-run overlap claim.')
    fig.tight_layout(rect=(0,.03,1,.97));save(fig,'figure_2_gpu_activity_evidence_levels')

    name=f'heldout_O16_N128_s{selection["arrival_seeds"][0]}_COST_v1';raw=ROOT/'raw'/name
    decisions=records(raw/'online_decisions.jsonl');start=json.loads((raw/'window.json').read_text())['start']
    fig,axes=plt.subplots(5,1,figsize=(14,12),sharex=True)
    times=[d['time']-start for d in decisions]
    axes[0].step(times,[d['choice']['layers'] for d in decisions],where='post');axes[0].set_ylabel('Selected layers')
    commits=records(raw/'actual_cut_commits.jsonl')
    axes[0].scatter([r['time']-start for r in commits],[r['offloaded_layers'] for r in commits],s=12,c='black',label='Actual committed cut')
    axes[0].legend()
    axes[1].step(times,[d['observation']['admitted_D_requests'] for d in decisions],where='post');axes[1].set_ylabel('Observed active D')
    budgets=[d['choice']['candidates'][0]['idle_budget_ms'] if d['choice']['candidates'] else math.nan for d in decisions]
    axes[2].plot(times,budgets,'o-');axes[2].set_ylabel('Predicted budget ms')
    axes[3].step(times,[d['observation']['P_backlog'] for d in decisions],where='post')
    axes[3].set_ylabel('Host P backlog')
    forecasts=json.loads((ROOT/'analysis/forecast_validation.json').read_text())['results']
    forecast=next((r for r in forecasts if r['name']==name),None)
    if forecast:
        rows=forecast['accepted_predictions'];axes[4].scatter([r['time']-start for r in rows],[r['completion_error_ms'] for r in rows])
    axes[4].axhline(0,color='black',lw=.7);axes[4].set_ylabel('First-token error ms');axes[4].set_xlabel('Seconds from observation start')
    fig.suptitle('Figure3 — Actual held-out COST decisions. Budget is a Host-state prediction, not measured GPU slack.')
    fig.tight_layout();save(fig,'figure_3_dynamic_cut_decisions')

    colors=dict(Native='#3b6fb6',Colocated='#d5852b',Static='#b44863',COST='#138e88',DEADLINE='#8553b6',IDLE='#768421',ControllerZero='#666666',MatchedFraction='#ce7098',FixedFraction50='#487c38')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for r in held:
        if r['cap']!=16:continue
        m=r['metrics'];color=colors[r['arm']]
        axes[0].scatter(m['TTFT_p95_ms']/1000,m['ITL_p99_ms'],color=color,s=30)
        axes[1].scatter(m['TTFT_p95_ms']/1000,m['TPOT_p95_ms'],color=color,s=30)
    for ax in axes:ax.set_xlabel('TTFT p95 seconds')
    axes[0].set_ylabel('ITL p99 ms');axes[1].set_ylabel('TPOT p95 ms')
    axes[0].legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=v,label=k) for k,v in colors.items()],fontsize=7)
    fig.suptitle('Figure4 — Held-out TTFT–Decode trade-off. Each point is an independent N128 start. FixedFraction50 is exploratory.')
    fig.tight_layout();save(fig,'figure_4_heldout_pareto')

    fig,axes=plt.subplots(3,2,figsize=(15,11))
    selected_stats=[r for r in stats if r['cap']==16]
    arms=[r['arm'] for r in selected_stats]
    for i,regime in enumerate(('TTFT_priority','Balanced','Decode_priority')):
        for j,key in enumerate(('Joint_attainment','Request_SLO_goodput')):
            ys=[r['SLO_regimes'][regime][key]['mean'] for r in selected_stats]
            axes[i,j].bar(arms,ys,color=[colors[a] for a in arms]);axes[i,j].tick_params(axis='x',rotation=35)
            axes[i,j].set_title(regime+' — '+key.replace('_',' '))
    starts=selected_stats[0]['independent_starts']
    fig.suptitle(f'Figure5 — Frozen SLO regimes. Means of{starts} starts. Finite-cohort goodput. FixedFraction50 is exploratory.')
    fig.tight_layout();save(fig,'figure_5_slo_attainment_goodput')

    fig,axes=plt.subplots(1,3,figsize=(15,5))
    ablations=[r for r in selected_stats if r['arm'] in ('Native','Static','MatchedFraction','FixedFraction50','ControllerZero','IDLE','DEADLINE','COST')]
    for ax,key in zip(axes,('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms')):
        ys=[r['ratios_vs_Native'][key]['ratio_of_means'] for r in ablations]
        bounds=[r['ratios_vs_Native'][key]['paired_bootstrap_95_percentile_CI'] for r in ablations]
        ax.bar([r['arm'] for r in ablations],ys,color=[colors[r['arm']] for r in ablations])
        ax.errorbar(range(len(ys)),ys,yerr=[[max(0,y-b[0]) for y,b in zip(ys,bounds)],
            [max(0,b[1]-y) for y,b in zip(ys,bounds)]],fmt='none',ecolor='black',capsize=3)
        ax.axhline(1,color='black',lw=.8)
        if key!='TTFT_p95_ms':ax.axhline(1.05,color='red',ls='--',lw=.8)
        ax.set_title(key.replace('_',' '));ax.tick_params(axis='x',rotation=40)
    fig.suptitle(f'Figure6 — Dynamic ablations. Paired95% bootstrap intervals. {starts} starts. FixedFraction50 is exploratory.')
    fig.tight_layout();save(fig,'figure_6_dynamic_ablation')
    write(ROOT/'figures/manifest.json',dict(files=[dict(path=str(p.relative_to(REPO)),SHA256=digest(p)) for p in sorted(folder.glob('*')) if p.suffix in ('.png','.svg')],
        scope='Figures use actual completed runs. Diagnostic kernels, stream enclosures, Host predictions, and clean serving are separate evidence levels. No synthetic speedup or idle value is plotted.'))
    print('FIGURES_READY')


if __name__=='__main__':main()

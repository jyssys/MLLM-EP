"""Plot measured operating points without inventing unmeasured kernel activity."""
import collections,csv,statistics
from .common import *
from .analyze import records


def completed_runs():
    values=[]
    for path in sorted((ROOT/'analysis').glob('*_metrics.json')):
        value=json.loads(path.read_text());name=value.get('name')
        if not name or not value.get('complete'):continue
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        if cfg.get('correctness_only') or not cfg.get('clean_timing',True):continue
        if cfg.get('execution_events') or cfg.get('full_prompt_state_witness'):continue
        if value.get('coalesced_intervals',0):continue
        gate=ROOT/'analysis'/(name+'_output_gate.json')
        if not gate.exists() or not json.loads(gate.read_text()).get('passed'):continue
        value['configuration']=cfg;values.append(value)
    return values


def category(cfg):
    arm=cfg['evaluation_arm']
    if arm=='Colocated':return 'Continuous Colocated'
    if arm in ('Native','NATIVE'):return 'Native / Controller Zero'
    if arm=='STATIC':return 'Static Layer PD'
    if arm=='DIRECT':return 'Deadline-aware Dynamic'
    return arm


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    runs=completed_runs();colors={'Native / Controller Zero':'#666666',
        'Continuous Colocated':'#0072b2','Static Layer PD':'#009e73','Deadline-aware Dynamic':'#d55e00'}
    caps=sorted({r['output_cap'] for r in runs if len(set(r['configuration']['request_caps']))==1})
    table=[]
    for r in runs:
        c=r['configuration']
        table.append(dict(name=r['name'],method=category(c),output_cap=r['output_cap'],
            offered_QPS=r['offered_QPS'],TTFT_p95_ms=r['TTFT_p95_ms'],ITL_p99_ms=r['ITL_p99_ms'],
            TPOT_p95_ms=r['TPOT_p95_ms'],raw_finite_cohort_output_tokens_s=r['output_tokens_s'],
            **{label+'_qualified_tokens_s':value['Output_token_SLO_goodput'] for label,value in r['SLO_regimes'].items()}))
    if table:
        with (ROOT/'analysis/finite_discovery_table.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    for cap in caps:
        chosen=[r for r in runs if r['output_cap']==cap and len(set(r['configuration']['request_caps']))==1
            and not r['configuration'].get('arrival_distribution') and not r['configuration'].get('fixed_measurement')]
        fig,axes=plt.subplots(1,3,figsize=(13,3.6))
        for method,color in colors.items():
            rows=[r for r in chosen if category(r['configuration'])==method and r['offered_QPS'] is not None]
            for ax,key,scale in zip(axes,('TTFT_p95_ms','TPOT_p95_ms','output_tokens_s'),(.001,1.,1.)):
                ax.scatter([r['offered_QPS'] for r in rows],[r[key]*scale for r in rows],
                    label=method,c=color,s=28,alpha=.7)
        for ax,label in zip(axes,('TTFT p95 (s)','Request TPOT p95 (ms)','Completed output tokens/s')):
            ax.set_xlabel('Offered requests/s');ax.set_ylabel(label);ax.grid(alpha=.25)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'Output{cap}: finite-cohort discovery, varied settings, no capacity claim')
        fig.tight_layout();fig.savefig(folder/f'finite_discovery_O{cap}.svg');plt.close(fig)
    long=[]
    for path in sorted((ROOT/'analysis').glob('*_sustainable_v2.json')):
        r=json.loads(path.read_text());cfg=json.loads((ROOT/'configs'/(r['name']+'.json')).read_text())
        if r.get('measurement_seconds',0)<600:continue
        long.append((r,cfg))
    for cap in sorted({c['cap'] for r,c in long}):
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        for ax,label in zip(axes,('TTFT_priority','Balanced','Decode_priority')):
            used=set()
            for r,c in long:
                if c['cap']!=cap:continue
                m=category(c);value=r['SLO_regimes'][label];color=colors.get(m,'black')
                ax.scatter(c['arrival_rate'],value['SLO_qualified_output_tokens_s'],s=55,
                    facecolors=color if value['qualified'] else 'none',edgecolors=color,
                    label=m if m not in used else None);used.add(m)
            ax.set_title(label);ax.set_xlabel('Offered requests/s');ax.grid(alpha=.25)
        axes[0].set_ylabel('SLO-qualified completed output tokens/s');axes[0].legend(fontsize=7)
        fig.suptitle(f'Output{cap}, measured windows >=600s: filled=qualified, hollow=rejected; no global maximum')
        fig.tight_layout();fig.savefig(folder/f'long_window_qualified_throughput_O{cap}.svg');plt.close(fig)
    cut_rows=[]
    for r in runs:
        c=r['configuration']
        if c.get('policy')!='DIRECT':continue
        counts=collections.Counter(row.get('completed_D_prefill_layers',0)
            for row in records(ROOT/'raw'/r['name']/'requests.jsonl'))
        cut_rows.append((r['name'],counts))
    if cut_rows:
        fig,ax=plt.subplots(figsize=(12,max(4.,.25*len(cut_rows))))
        for i,(name,counts) in enumerate(cut_rows):
            left=0
            for layers,color in zip((0,2,4,8,12),('#cccccc','#56b4e9','#009e73','#e69f00','#d55e00')):
                fraction=counts[layers]/sum(counts.values());ax.barh(i,fraction,left=left,color=color,
                    label=f'Last{layers}' if i==0 else None);left+=fraction
        ax.set_yticks(range(len(cut_rows)),[v[0] for v in cut_rows],fontsize=6)
        ax.set_xlabel('Actual completed request fraction');ax.set_xlim(0,1);ax.legend(ncol=5,fontsize=8)
        ax.set_title('Actual online choices; forced qualifications excluded')
        fig.tight_layout();fig.savefig(folder/'dynamic_actual_cut_distribution.svg');plt.close(fig)
    write(ROOT/'analysis/figure_provenance.json',dict(time=time.time(),finite_runs=[r['name'] for r in runs],
        long_window_runs=[r['name'] for r,c in long],evidence='MEASURED',
        physical_kernel_activity='NOT_MEASURED',
        scope='Scatter points preserve different settings. They are not interpolated capacity curves. Final independent-start confidence intervals require separately frozen confirmation selection.'))
    print('FIGURES',len(runs),len(long),flush=True)


if __name__=='__main__':main()

"""Plot only qualified Main measurements with server-start confidence intervals."""
import itertools
import json
import numpy as np
from .common import ROOT,REPO,write
from .metrics import summarize


def start_interval(values):
    if len(values)<2:return None
    means=[np.mean([values[i] for i in draw]) for draw in itertools.product(range(len(values)),repeat=len(values))]
    return [float(np.percentile(means,p)) for p in (2.5,97.5)]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for folder in sorted((ROOT/'raw').glob('Main_*_LongBench_N200_*')):
        receipt=folder/'dispatcher_result.json';gate=ROOT/'analysis'/(folder.name+'_output_gate.json')
        if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
        if not gate.exists() or not json.loads(gate.read_text())['passed']:continue
        cfg=json.loads((folder/'configuration.json').read_text())
        if (cfg['token_budget'],cfg['max_num_seqs'])!=(8192,1024):continue
        rows.append(summarize(folder.name))
    keys=[('ttft_p95','TTFT_p95_ms','TTFT p95 (ms)'),
        ('itl_p99','ITL_p99_ms','Client-visible ITL p99 (ms)'),
        ('tpot_p95','TPOT_p95_ms','Request TPOT p95 (ms/token)'),
        ('output_tokens_s','generated_output_tokens_per_second','Generated output tokens/s'),
        ('SLO_attainment','Joint_SLO_attainment','Joint SLO attainment (fraction)'),
        ('SLO_goodput','SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s')]
    assert rows,'Do not create blank Main curves from missing or failed results'
    complete=(ROOT/'state/main_common_sweep_complete.json').exists()
    table=[]
    for stem,key,ylabel in keys:
        fig,ax=plt.subplots(figsize=(7.4,5.1))
        for method,color,marker,label in (('Agg','#d95f02','o','Agg DP2'),('Native','#009e73','^','Native PD 1P1D')):
            part=[r for r in rows if r['method']==method];xs=sorted({r['offered_QPS'] for r in part});ys=[];low=[];high=[]
            for rate in xs:
                values=[r[key] for r in part if r['offered_QPS']==rate];mean=float(np.mean(values));ci=start_interval(values)
                ys.append(mean);low.append(ci[0] if ci else np.nan);high.append(ci[1] if ci else np.nan)
                table.append(dict(figure=stem,method=method,QPS=rate,mean=mean,CI95=ci,independent_starts=len(values)))
            ax.plot(xs,ys,marker=marker,color=color,label=label)
            if np.isfinite(low).any():ax.errorbar(xs,ys,yerr=[np.asarray(ys)-np.asarray(low),np.asarray(high)-np.asarray(ys)],fmt='none',color=color,capsize=3)
        ax.set(xlabel='Offered QPS (requests/s)',ylabel=ylabel)
        ax.set_title('Qwen3-8B | 2×H100 | Frozen LongBench v1 Mixed200\nB8192/S1024 | Natural EOS, cap2048 | Official batch-invariant path\n'+('Complete frozen-grid study' if complete else 'IN PROGRESS: current independent starts only'),fontsize=10)
        ax.grid(alpha=.2);ax.legend()
        counts='Starts/method: n=3 at QPS1/1.65; n=1 elsewhere.' if complete else 'Starts are incomplete. Per-cell counts are in the statistics JSON.'
        tokens=sorted({r['output_tokens'] for r in rows})
        work=f'200 requests/start. Actual output tokens/start: {tokens[0]:,}.' if len(tokens)==1 else 'Actual output totals are in the statistics JSON.'
        fig.text(.03,.012,counts+' '+work+'\nFinite-cohort throughput, not sustainable capacity. 95% CI: server-start bootstrap. n=1 has no CI.',fontsize=7)
        fig.tight_layout(rect=(0,.065,1,1))
        for suffix in ('png','pdf'):
            fig.savefig(ROOT/'figures'/f'longbench200_{stem}_vs_qps.{suffix}',dpi=180)
            fig.savefig(REPO/'poc_flashvep'/f'longbench200_{stem}_vs_qps.{suffix}',dpi=180)
        plt.close(fig)
    write(ROOT/'analysis/Main_figure_start_level_statistics.json',dict(rows=table,
        complete=complete,evidence='MEASURED_ONLINE',scope='Only output- and lifetime-qualified Main jobs. One-server-start cells have no confidence interval. Requests are not independent starts.'))
    write(ROOT/'analysis/Main_per_start_metrics.json',dict(rows=rows))


if __name__=='__main__':main()

"""Plot the validated longer-prompt search without inventing unmeasured DP rates."""
import collections
from .common import *
from .analyze import table


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs=json.loads((ROOT/'serving_runs.json').read_text())
    manifest=json.loads((ROOT/'figure_manifest.json').read_text());extract=[]
    for model,length in [('ouro',8192),('huginn',3072)]:
        groups=collections.defaultdict(list)
        for r in runs:
            if r['model']!=model or r['length']!=length or r['status']!='COMPLETE':continue
            c=json.loads((ROOT/'configs'/(r['name']+'.json')).read_text())
            if not c['name'].startswith('warm_secondary_') or c.get('phase') not in ('qps_pilot','aggregated_QPS_pilot'):continue
            label=('DP2 Native Mixed ' if c['arm'].startswith('DP') else 'PD ')+('Atomic' if c['atomic_prompt'] else 'C'+str(c['chunk']))
            groups[label].append(r);extract.append(dict(figure=model+'_longer_prompt_search',series=label,name=r['name'],
                length=length,cap=c['cap'],target_QPS=c['arrival_rate'],TTFT_ms_p95=r['TTFT_ms_p95'],TPOT_ms_p95=r['request_TPOT_ms_p95'],
                ITL_ms_p99=r['token_ITL_ms_p99'],total_tokens_per_s=r['total_tokens_per_s']))
        if not groups:continue
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for label,values in sorted(groups.items()):
            values.sort(key=lambda r:r['target_QPS'])
            for axis,field in zip(axes,('TTFT_ms_p95','token_ITL_ms_p99','total_tokens_per_s')):
                axis.plot([r['target_QPS'] for r in values],[r[field] for r in values],
                    'o' if label.startswith('DP2') else 'o-',label=label)
        for axis,y in zip(axes,('TTFT p95 (ms)','Token ITL p99 (ms)','Input + output tokens/s')):
            axis.set(xlabel='Target offered requests/s',ylabel=y);axis.grid(alpha=.2);axis.set_ylim(bottom=0)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{model.title()} | {length}/32 | N=24 exploratory starts\nDP2 dots exist only at the measured matching rate. No line estimates an unmeasured DP2 rate.',fontsize=10)
        fig.tight_layout();outputs=[]
        for ext in ('png','svg'):
            p=ROOT/'figures'/(model+'_longer_prompt_search.'+ext);fig.savefig(p,dpi=180);outputs.append(str(p.relative_to(ROOT)))
        plt.close(fig)
        manifest['figures'].append(dict(name=model+'_longer_prompt_search',outputs=outputs,
            exact_sources=['serving_runs.json','longer_prompt_figure_data.csv'],
            scope='Actual warmed longer-prompt pilots. PD chunk budgets receive three rates. Each DP2 chunk budget receives the selected same offered rate. No interpolation at missing DP points.'))
    table('longer_prompt_figure_data.csv',extract)
    write(ROOT/'figure_manifest.json',manifest)


if __name__=='__main__':main()

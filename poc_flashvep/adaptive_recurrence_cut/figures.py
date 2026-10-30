"""Plot actual finite-cohort observations. Never label these as serving capacity."""
import csv
from collections import defaultdict
import statistics
from .common import ROOT


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((ROOT/'static_cut_metrics_by_start.csv').open()))
    for model in ('ouro','huginn'):
        selected=[r for r in rows if r['model']==model and int(r['prompt_tokens'])==512 and int(r['start'])<3 and r['regime']!='D_pressure' and r['handoff_mode']=='cut_snapshot']
        regimes=['P_pressure','balanced','verified_D_pressure']
        cuts=sorted({int(r['cut']) for r in selected})
        fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
        for ax,field,title in zip(axes,('ttft_ms_mean','tpot_ms_mean','all_requests_e2e_ms_mean'),
                                 ('Mean target TTFT (ms)','Mean target TPOT (ms)','Mean all-request E2E (ms)')):
            values=[]
            for regime in regimes:
                line=[]
                for cut in cuts:
                    actual=[float(r[field]) for r in selected if r['regime']==regime and int(r['cut'])==cut]
                    line.append(statistics.mean(actual) if actual else float('nan'))
                values.append(line)
            image=ax.imshow(values,aspect='auto',cmap='viridis')
            ax.set_xticks(range(len(cuts)),cuts);ax.set_yticks(range(3),regimes);ax.set_xlabel('P-side recurrent cut')
            ax.set_title(title);fig.colorbar(image,ax=ax,shrink=.8)
        starts=len({r['start'] for r in selected})
        fig.suptitle(f'{model}: measured L512, chunk128, fixed full depth, Mixed OFF\n8 requests/cell/start; {starts} starts available; verified-D is state-triggered; not capacity')
        fig.savefig(ROOT/'figures'/f'{model}_static_cut_heatmaps.png',dpi=180)
        fig.savefig(ROOT/'figures'/f'{model}_static_cut_heatmaps.svg');plt.close(fig)
        fig,ax=plt.subplots(figsize=(6.8,4.5),layout='constrained')
        colors={0:'tab:gray',1:'tab:blue',2:'tab:orange',3:'tab:green'}
        for regime in regimes:
            for cut in cuts:
                group=[r for r in selected if r['regime']==regime and int(r['cut'])==cut]
                if not group:continue
                x=statistics.mean(float(r['ttft_ms_mean']) for r in group)
                y=statistics.mean(float(r['tpot_ms_mean']) for r in group)
                ax.scatter(x,y,marker={'P_pressure':'o','balanced':'s','verified_D_pressure':'^'}[regime])
                ax.annotate(f'{cut}/{regime[0]}',(x,y),fontsize=8)
        ax.set(xlabel='Mean target TTFT (ms)',ylabel='Mean target TPOT (ms)',
               title=f'{model}: observed fixed-cut trade-off\nL512/chunk128; eight-request cohorts; no capacity claim')
        fig.savefig(ROOT/'figures'/f'{model}_ttft_tpot_tradeoff.png',dpi=180)
        fig.savefig(ROOT/'figures'/f'{model}_ttft_tpot_tradeoff.svg');plt.close(fig)


if __name__=='__main__':main()

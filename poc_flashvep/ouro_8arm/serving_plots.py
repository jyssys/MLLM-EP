"""Plot measured prepared-service ratios and their preparation-inclusive controls."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .contract import *


def main():
    file=ROOT/'replay_summary.csv'
    if not file.exists():return
    rows=list(csv.DictReader(file.open()));out=ROOT/'figures';out.mkdir(exist_ok=True)
    for family,arms,name in [('STATIC_FROZEN_TOKENS',['S4','T4','L4','O4'],'static'),
            ('TRACE_LOCKED_DYNAMIC_TOKENS',['SD','TD','LD','OD'],'dynamic')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for arm in arms:
            selected=[r for r in rows if r['arm']==arm and r['family']==family and
                (name=='dynamic' or r['case'] in ['0','1','2','3'])]
            selected.sort(key=lambda r:int(r['C']))
            selected=[r for r in selected if r['speedup_cohort_vs_mode_Single']]
            if selected:ax.plot([int(r['C']) for r in selected],[float(r['speedup_cohort_vs_mode_Single']) for r in selected],'o-',label=arm)
        ax.axhline(1.,color='gray',ls=':');ax.set(xlabel='Concurrent requests',ylabel='Prepared-service cohort speedup (x)',
            title=name.capitalize()+' identical-work replay. TP is numerically exploratory.')
        if ax.lines:ax.legend()
        fig.tight_layout();fig.savefig(out/(name+'_prepared_service_scaling.png'),dpi=160);plt.close(fig)
    selected=[r for arm in ['S4','T4','L4','O4'] for r in rows if r['arm']==arm and r['family']=='STATIC_FROZEN_TOKENS' and r['case']=='1']
    if selected:
        fig,ax=plt.subplots(figsize=(8,4));x=list(range(len(selected)))
        ax.bar([v-.18 for v in x],[float(r['median_cohort_makespan_s']) for r in selected],width=.36,label='Prepared service only')
        available=[(i,r) for i,r in enumerate(selected) if r['resident_model_cohort_median_s']]
        ax.bar([i+.18 for i,r in available],[float(r['resident_model_cohort_median_s']) for i,r in available],width=.36,label='Includes per-cohort preparation')
        ax.set(xticks=x,xticklabels=[r['arm'] for r in selected],ylabel='Median cohort wall time (s)',
            title='C4/P128/O32. Resident model. Preparation is not free.')
        ax.legend();fig.tight_layout();fig.savefig(out/'graph_preparation_vs_service.png',dpi=160);plt.close(fig)
    print('SERVING_FIGURES',len(rows),'measured replay rows',flush=True)


if __name__=='__main__':main()

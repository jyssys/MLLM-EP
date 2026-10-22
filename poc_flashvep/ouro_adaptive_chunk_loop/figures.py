"""Plot actual contention controls. Missing measurements remain absent."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write

def main():
    path=ROOT/'contention_paired_comparisons.csv'
    rows=list(csv.DictReader(path.open()))
    out=ROOT/'figures'
    out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,p,n in zip(axes,[2048,8192],[4,8]):
        for gpu in range(8):
            points=sorted([r for r in rows if int(r['GPU'])==gpu and int(r['P'])==p and r['metric']=='median_ITL_ms'],key=lambda r:int(r['parallel_workers']))
            ax.plot([1]+[int(r['parallel_workers']) for r in points],[0]+[float(r['percent_change']) for r in points],marker='o',label=f'GPU{gpu}')
        ax.axhline(3,color='black',linestyle='--',linewidth=1,label='3% review trigger')
        ax.set(xlabel='Simultaneous independent GPU workers',ylabel='Same-GPU median-ITL drift versus solo (%)',title=f'P≈{p}, N={n} — warmed calibration',xticks=[1,2,4,8])
        ax.grid(alpha=.25)
    summary=json.loads((ROOT/'contention_summary.json').read_text())
    fig.suptitle(f"LIVE calibration: {summary['completed_jobs']}/{summary['registered_jobs']} completed jobs. Not a policy speedup.")
    axes[1].legend(fontsize=8,ncol=2)
    for suffix in ['png','svg']:
        fig.savefig(out/f'contention_device_crossover.{suffix}',dpi=180)
    plt.close(fig)
    write(out/'contention_figure_manifest.json',dict(source=str(path.relative_to(ROOT)),evidence='LIVE_MEASURED_CALIBRATION',
        completed_jobs=summary['completed_jobs'],scope='Matched drift of cohort medians. Device lines are not independent request samples.'))

if __name__=='__main__':
    main()

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *


def main():
    data=json.loads((ROOT/'analysis/output_idle_diagnosis.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,7));manifest=[]
    for i,row in enumerate(data['rows']):
        name=row['names']['capture'];ax=axes[i,0];hist=axes[i,1]
        if row['capture'] is None:
            for panel in (ax,hist):
                panel.text(.5,.5,'Output '+str(row['output_cap'])+'\nNOT_MEASURED\nCUDA launch failure during capture',ha='center',va='center',transform=panel.transAxes)
                panel.set_axis_off()
            continue
        origin=row['capture']['decode']['window'][0]
        for y,role in enumerate(('prefill','decode')):
            with (ROOT/'analysis'/(name+'_'+role+'_kernel_union.csv')).open() as f:intervals=list(csv.DictReader(f))
            ax.broken_barh([(1000*(float(r['start'])-origin),1000*(float(r['end'])-float(r['start']))) for r in intervals],
                (y-.3,.6),facecolors='#d5a041' if role=='prefill' else '#447e9f')
        ax.set_yticks([0,1],['P kernels','D kernels']);ax.set_xlim(0,1000*row['capture']['decode']['window_seconds'])
        ax.set_title(f'Output {row["output_cap"]}: captured kernel union')
        ax.set_xlabel('ms after common capture start')
        with (ROOT/'analysis'/(name+'_decode_idle_windows.csv')).open() as f:windows=list(csv.DictReader(f))
        durations=[float(r['duration_ms']) for r in windows]
        hist.hist(durations,bins=[0,.001,.002,.005,.01,.02,.05,.1,.2,.5,1,2,5,10,20,30,50,100,200,500,1000])
        hist.set_xscale('symlog',linthresh=.001);hist.set_yscale('log')
        hist.set_xlabel('D kernel-idle window, ms');hist.set_ylabel('Count')
        label='passed point screen only' if row['strict_capture_qualified'] else 'FAILED sensitivity screen: diagnostic only'
        hist.set_title(label,fontsize=10)
    fig.suptitle('Actual instrumented kernel intervals. Not measured Decode-safe slack.\nFive-second common windows, not whole-cohort idle totals. Copy-engine activity is not shown.')
    fig.tight_layout(rect=(0,0,1,.91));out=ROOT/'figures/output_cap_kernel_idle_diagnostic.png';fig.savefig(out,dpi=150);plt.close(fig)
    manifest.append(dict(path=str(out.relative_to(ROOT)),scope='Actual captured interval union. Separate profiler screens determine whether clean interpretation is allowed.'))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,entry in zip(axes,data['forecasts']):
        path=ROOT/'analysis'/(entry['name']+'_forecast_errors.csv')
        if not path.exists():ax.text(.5,.5,'NOT_MEASURED',ha='center');continue
        with path.open() as f:rows=list(csv.DictReader(f))
        ax.scatter([float(r['prediction_ms']) for r in rows],[float(r['actual_known_cohort_drain_ms']) for r in rows],s=5,alpha=.4)
        maximum=max([float(r['prediction_ms']) for r in rows]+[float(r['actual_known_cohort_drain_ms']) for r in rows])
        ax.plot([0,maximum],[0,maximum],color='gray',linestyle=':');ax.set_xlabel('Advertised known-cohort prediction, ms');ax.set_ylabel('Actual known-cohort drain, ms')
        ax.set_title(entry['name'].replace('idle_drain_','').replace('_audit_n64',''))
    fig.suptitle('Host forecast audit. Points are correlated advertisements, not independent requests.')
    fig.tight_layout(rect=(0,0,1,.91));out=ROOT/'figures/known_cohort_drain_forecast.png';fig.savefig(out,dpi=150);plt.close(fig)
    manifest.append(dict(path=str(out.relative_to(ROOT)),scope='Measured Host output-send completion against actual advertised predictions. Not GPU idle or a deadline guarantee.'))
    write(ROOT/'output_figure_manifest.json',manifest)


if __name__=='__main__':main()

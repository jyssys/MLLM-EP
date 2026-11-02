"""Display measured start-level variation and paired 95% uncertainty."""
import csv
import statistics
from .common import *
from .paired_analysis import interval


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs=list(csv.DictReader((ROOT/'paired_repetitions.csv').open()))
    manifest=json.loads((ROOT/'figure_manifest.json').read_text())
    if not pairs:return
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    fields=[('paired_goodput_difference_PD_minus_DP','PD minus DP2 SLO goodput (req/s)'),
        ('paired_TTFT_p95_difference_ms','PD minus DP2 TTFT p95 (ms)'),
        ('paired_ITL_p99_difference_ms','PD minus DP2 token ITL p99 (ms)')]
    for axis,(field,label) in zip(axes,fields):
        for pos,model in enumerate(('ouro','huginn')):
            values=[float(r[field]) for r in pairs if r['model']==model]
            if not values:continue
            axis.scatter([pos+(i-(len(values)-1)/2)*.06 for i in range(len(values))],values,s=25,label=model.title()+' starts')
            mean=statistics.mean(values);lo,hi=interval(values)
            if lo is not None:axis.errorbar([pos],[mean],yerr=[[mean-lo],[hi-mean]],fmt='ks',capsize=5)
            axis.text(pos,.97,f'n={len(values)} starts',ha='center',va='top',transform=axis.get_xaxis_transform(),fontsize=8)
        axis.axhline(0,color='black',ls='--',lw=.8);axis.set_xticks([0,1],['Ouro','Huginn'])
        axis.set_ylabel(label);axis.grid(alpha=.2)
    fig.suptitle('Matched PD / Native Mixed DP2 repetitions | N=128 new prompts per start\nPoints: paired independent starts. Black: mean and Student-t 95% CI when all three starts complete.',fontsize=10)
    fig.tight_layout();outputs=[]
    for ext in ('png','svg'):
        path=ROOT/'figures'/('paired_start_uncertainty.'+ext);fig.savefig(path,dpi=180);outputs.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    manifest['figures'].append(dict(name='paired_start_uncertainty',outputs=outputs,
        exact_sources=['paired_repetitions.csv','paired_start_uncertainty.csv'],
        source_hashes={name:digest(ROOT/name) for name in ('paired_repetitions.csv','paired_start_uncertainty.csv')},
        scope='Start-level paired differences. Three starts support only a wide df=2 interval. No token-level pseudoreplication. Not a sustainable-capacity claim.'))
    write(ROOT/'figure_manifest.json',manifest)


if __name__=='__main__':main()
